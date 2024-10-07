import gradio as gr
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import os
import tempfile
from worker_deeplx import translate_text

from typing import Any, List, NamedTuple, Optional, Tuple, Union

from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import torch

global_outputs = ["请等待管理员加载", "刘青沅在忙", "别催了，别说不好用了，服务器不要你钱已经不错了", "要新功能可以找我，反正我没时间改U•ェ•*U"]

class Punctuator:
    def __init__(self, model_name="Qishuai/distilbert_punctuator_en"):
        """初始化 Punctuator 类，加载模型和分词器。

        参数：
            model_name (str): 使用的模型名称，默认为 "Qishuai/distilbert_punctuator_en"。
        """
        self.model = DistilBertForTokenClassification.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        
        # 设置模型为评估模式
        self.model.eval()

    def add_punctuation(self, text: str) -> str:
        """给输入文本添加标点符号。

        参数：
            text (str): 待处理的输入文本。

        返回：
            str: 添加了标点符号的文本。
        """
        # 对输入文本进行分词
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs).logits

        # 获取预测的标签
        predictions = torch.argmax(outputs, dim=2)

        # 获取输入文本的tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # 获取预测的标签
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]

        # 生成带标点的文本
        result_text = ""
        for token, label in zip(tokens, labels):
            # print("token:", token)
            # print("label:", label)
            # 忽略 [CLS] 和 [SEP] 标记
            if token in ['[CLS]', '[SEP]']:
                continue

            # 处理特殊标记
            if token.startswith('##'):
                token = token[2:]
                if result_text and result_text[-1] == ' ':
                    result_text = result_text[:-1] + token + ' '
                else:
                    result_text += token + ' '
            # 添加标点符号
            elif label == 'O':
                if token in ["'", '"', '`', '-', '—', '--', '_']:
                    if result_text and result_text[-1] == ' ':
                        result_text = result_text[:-1] + token
                    else:
                        result_text += token
                elif token in [',', '.', '!', '?', ';', ':']:
                    if result_text and result_text[-1] == ' ':
                        result_text = result_text[:-1] + token + ' '
                    else:
                        result_text += token + ' '
                else:
                    result_text += token + ' '
            # elif label == 'COMMA':
            #     result_text += ', '
            # elif label == 'PERIOD':
            #     result_text += '. '
            
        return result_text.strip()



# 定义 Word 类
class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float

# 定义 Segment 类
class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]
    temperature: Optional[float] = 1.0

class TextSegment:
    def __init__(self, show_text: str, global_end_time: float):
        self.show_text = show_text
        self.global_end_time = global_end_time

class Transcriber:
    def __init__(self):
        self.model_size = "large-v3" # tiny, tiny.en, base, base.en,small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,large-v2, large-v3, large, distil-large-v2 or distil-large-v3
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
        self.temp_dir = "/home/24052432g/work/temp"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.buffers = {3: np.array([], dtype=np.float32), 30: np.array([], dtype=np.float32), 60: np.array([], dtype=np.float32)}
        self.show_texts = {3: "", 30: "", 60: ""}
        self.global_buffer_times = {3: 0, 30: 0, 60: 0}
        self.last_if_seg_buffer_times = {3: 0, 30: 0, 60: 0}
        self.if_seg_buffer_time_secs = {3: 3, 30: 30, 60: 60}
        self.segments_lists = {3: [], 30: [], 60: []}
        self.seg_end_time_tolerances = {3: 0.00, 30: 0.05, 60: 0.00}
        self.merge_tolerances = { 3: 0,30: 0.01,60: 0.01}
        self.text_segments = {3: [], 30: [], 60: []}

    def clear_temp_dir(self) -> None:
        """清空临时文件夹中的所有文件。

        返回：
            None
        """
        for file_name in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file_name)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    def clear_temp_file(self, temp_filename: str) -> None:
        """删除指定的临时文件。

        参数：
            temp_filename (str): 临时文件的路径。

        返回：
            None
        """
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"临时 WAV 文件已删除: {temp_filename}")

    def save_temp_wav(self, data: np.ndarray, sr: int) -> str:
        """保存音频数据到临时 WAV 文件并返回文件路径。

        参数：
            data (np.ndarray): 要保存的音频数据。
            sr (int): 音频采样率。

        返回：
            str: 临时 WAV 文件的路径。
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.temp_dir) as temp_wav_file:
            # 将数据写入 WAV 文件
            sf.write(temp_wav_file.name, data, sr)
            # print(f"临时 WAV 文件已创建: {temp_wav_file.name}")
        return temp_wav_file.name

    def process_audio_data(self, y: np.ndarray) -> np.ndarray:
        """处理音频数据，包括立体声转换和归一化。

        参数：
            y (np.ndarray): 输入音频数据。

        返回：
            np.ndarray: 处理后的音频数据。
        """
        if y.ndim == 2 and y.shape[1] == 2:
            y = np.mean(y, axis=1)  # 将立体声转换为单声道
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        if np.max(np.abs(y)) > 1.0:  # 如果需要，归一化值
            y = y / np.max(np.abs(y))
        return y

    def get_combined_text(self) -> str:
        """综合生成翻译文本。

        返回：
            str: 综合所有有效段落的翻译文本。按照优先级顺序输出 60 秒、30 秒、3 秒的段落。
                只有当段落的 global_end_time 超过已输出的最大值时才会被添加到最终文本中。
                同时，未输出的段落会被释放以节省内存。
        """
        combined_text = ""
        max_global_end_time = 0  # 跟踪已输出的最大 global_end_time

        # 按照优先级顺序输出 60 秒、30 秒、3 秒的段落
        for buffer_time in [60, 30, 3]:
            new_segments = []
            for segment in self.text_segments[buffer_time]:
                # 只有在 global_end_time 超过当前最大值时才输出
                if segment.global_end_time > max_global_end_time:
                    combined_text += segment.show_text
                    max_global_end_time = segment.global_end_time  # 更新最大 global_end_time
                    new_segments.append(segment) # 保留输出的segment 删除未输出的segment

            # 更新 text_segments 以释放不需要的段落
            self.text_segments[buffer_time] = new_segments
        return combined_text

    def generate_translation(self, segments: List[Segment], buffer_time: int, max_show_text_length: int = 3000000) -> str:
        """生成翻译文本。

        参数：
            segments (List[Segment]): 待翻译的音频片段列表。
            buffer_time (int):对应buffer时间。
            max_show_text_length (int): 最大显示文本长度，超出部分将被截断。

        返回：
            str: 生成的翻译文本。
        """
        show_text = self.show_texts[buffer_time]
        global_start_time = self.global_buffer_times[buffer_time]  # 获取全局开始时间

        if segments:
            for segment in segments:
                # 更新全局时间
                global_start_time = self.global_buffer_times[buffer_time] + segment.start
                global_end_time = self.global_buffer_times[buffer_time] + segment.end 
                
                punctuator = Punctuator()
                p_text = punctuator.add_punctuation(segment.text)

                s_text = f"[{global_start_time:.2f}s -> {global_end_time:.2f}s] {p_text}\n"
                trans_text = translate_text(p_text, "en", "zh")  # 使用翻译API函数
                s_trans = f"[{global_start_time:.2f}s -> {global_end_time:.2f}s] {trans_text}\n"

                t_text = s_text + s_trans
                show_text += t_text

                new_segment = TextSegment(t_text, global_end_time)
                self.text_segments[buffer_time].append(new_segment)

            self.global_buffer_times[buffer_time] = global_end_time  # 累加全局时间

            if len(show_text) > max_show_text_length:
                show_text = show_text[-max_show_text_length:]
            # print(show_text)
        return show_text

    def merge_segments(self, segments: List[Segment], tolerance: float = 0.01) -> List[Segment]:
        """合并相邻的 Segment 对象。

        参数：
            segments (List[Segment]): 要合并的音频片段列表。
            tolerance (float): 合并时允许的时间戳容忍度。

        返回：
            List[Segment]: 合并后的音频片段列表。
        """
        if not segments:
            return []

        merged_segments = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            # 检查当前段与下一个段之间的时间戳差异
            if next_segment.start - current_segment.end < tolerance:
                # 合并相邻的段
                current_segment = Segment(
                    id=current_segment.id,
                    seek=current_segment.seek,
                    start=current_segment.start,
                    end=next_segment.end,
                    text=current_segment.text + " " + next_segment.text,
                    tokens=current_segment.tokens + next_segment.tokens,
                    avg_logprob=(current_segment.avg_logprob + next_segment.avg_logprob) / 2, # 平均
                    compression_ratio=(current_segment.compression_ratio + next_segment.compression_ratio) / 2, # 平均
                    no_speech_prob=(current_segment.no_speech_prob + next_segment.no_speech_prob) / 2,  # 取平均
                    words=(current_segment.words or []) + (next_segment.words or []),  # 确保是列表
                    temperature=current_segment.temperature
                )
            else:
                # 如果不在容忍度内，保存当前段并继续
                merged_segments.append(current_segment)
                current_segment = next_segment

        # 添加最后一个段
        merged_segments.append(current_segment)
        
        return merged_segments


    def buffer_gen(self, y: np.ndarray, sr: int, buffer_time: int) -> Optional[np.ndarray]:
        """处理音频缓冲区。

        参数：
            y (np.ndarray): 新增的音频数据。
            sr (int): 音频采样率。
            buffer_time (int):对应buffer时间。

        返回：
            np.ndarray or None: 处理完成的音频数据或 None。
        """

        # 更新缓冲区
        self.buffers[buffer_time] = np.concatenate((self.buffers[buffer_time], y))  # 直接连接新的音频数据
        buffer_duration = self.buffers[buffer_time].shape[0] / sr  # 根据 buffer 的长度计算时长
        print(f"当前缓冲区时长: {buffer_duration:.2f} 秒")

        # 处理buffer（当buffer_duration达到阈值时）
        if buffer_duration >= self.last_if_seg_buffer_times[buffer_time] + self.if_seg_buffer_time_secs[buffer_time]:
            # 更新最后处理时间
            self.last_if_seg_buffer_times[buffer_time] += self.if_seg_buffer_time_secs[buffer_time]
            temp_buffer = self.buffers[buffer_time].copy() # 复制当前 buffer 进行处理

            temp_filename = self.save_temp_wav(temp_buffer, sr)
            
            # 识别缓冲区
            segments, info = self.transcribe_audio(temp_filename)
            # print("info:", info)
            self.clear_temp_file(temp_filename)

            # 处理识别结果(是否切割)
            # 缓冲策略：
            # 识别到speech
            if segments:
                seg_end_time = 0  # 初始化最后一个段的结束时间
                self.segments_lists[buffer_time] = list(segments)  # 复制为列表

                
                if self.segments_lists[buffer_time]:  # 检查 segments_list 是否为空
                    seg_end_time = self.segments_lists[buffer_time][-1].end  # 直接获取最后一个段的结束时间
                    # for segment in segments_list:
                    #     s_text = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                    #     print(s_text)
                    print(f"最后一个段的结束时间: {seg_end_time:.2f} 秒")
                else:
                    #     # 抛出异常，带有提示信息
                    #     raise ValueError("Unexpected behavior: segments_list is empty when it should contain segments.")
                    # self.buffers[buffer_time] = np.array([], dtype=np.float32) # 清空 buffer
                    return None
                
                
                # buffer结尾里是speech
                if seg_end_time == buffer_duration or abs(seg_end_time - buffer_duration) <= self.seg_end_time_tolerances[buffer_time]: 
                    # 去掉最后一个元素，保留倒数第二个及之前的元素
                    if len(self.segments_lists[buffer_time]) > 1:
                        # 去掉最后一个元素
                        self.segments_lists[buffer_time] = self.segments_lists[buffer_time][:-1]  
                        seg_end_time = self.segments_lists[buffer_time][-1].end  # 获取倒数第二段的结束时间
                    else:
                        # 如果 segments_list 中只有一段，直接清空列表->基本上buffer里全是speech，跳过输出，并切割保留全部
                        self.segments_lists[buffer_time] = []
                        seg_end_time = 0
                    # return None
                
                # 判断是否需要切割
                print("buffer_duration - seg_end_time:", buffer_duration - seg_end_time)
                if abs(seg_end_time - buffer_duration) > self.seg_end_time_tolerances[buffer_time]:
                    seg_end_samples = int(seg_end_time * sr)
                    self.buffers[buffer_time] = temp_buffer[seg_end_samples:] # 更新 buffer 为未处理的数据部分
                    self.last_if_seg_buffer_times[buffer_time] = 0
                    return temp_buffer[:seg_end_samples] # 处理完成的数据
            # 没有识别到speech直接清空buffer
            # self.buffers[buffer_time] = np.array([], dtype=np.float32) # 清空 buffer
        return None

    def transcribe_audio(self, file_path: str) -> Tuple[List[Segment], Any]:
        """对音频文件进行转录。

        参数：
            file_path (str): 音频文件的路径。

        返回：
            Tuple[List[Segment], Any]: 转录结果的音频片段列表和相关信息。
        """
        try:
            segments, info = self.model.transcribe(file_path, beam_size=5, length_penalty=1, 
                                                    repetition_penalty=1.5, # 重复惩罚
                                                    chunk_length=30, # The length of audio segments.
                                                    language="en",
                                                    vad_filter=True,
                                                    vad_parameters=dict(
                                                        threshold= 0.5,
                                                        min_speech_duration_ms = 250,
                                                        max_speech_duration_s = float("inf"),
                                                        min_silence_duration_ms = 200,
                                                        speech_pad_ms = 600
                                                        )
                                                    )
            # print(f"转录完成，获得 {len(segments)} 个 segments")
            return segments, info
        except RuntimeError as e:
            if "stack expects a non-empty TensorList" in str(e):
                print("Empty TensorList encountered, skipping this transcription.")
            else:
                raise
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def transcribe(self, stream, new_chunk, password: Tuple[int, np.ndarray]) -> Tuple:
        """主转录函数。

        参数：
            stream: 输入音频流。
            new_chunk: 新接收到的音频数据。

        返回：
            Tuple: 包含音频流和生成的文本。
        """
        global global_outputs
        # 检查密码
        if password != 1024:
            return stream, *global_outputs
        if new_chunk is None:
            return stream, "Error: No audio received.", "Error: No audio received.", "Error: No audio received.", "Error: No audio received."

        sr, y = new_chunk

        # 处理立体声和数据类型
        y = self.process_audio_data(y)

        global_outputs = []
        for buffer_time in [3, 30, 60]:
            # 查看temp_buffer是否为空，不为空则输出当时缓冲区模型的识别结果segments_list
            temp_buffer = self.buffer_gen(y, sr, buffer_time)
            if isinstance(temp_buffer, np.ndarray):
                temp_buffer_duration = temp_buffer.shape[0] / sr
                print(f"处理buffer_time={buffer_time}的缓冲区中 {temp_buffer_duration} 秒的样本")
                # 使用保留的 segments_list 进行文本处理
                if self.segments_lists[buffer_time]:
                    merged_segments = self.merge_segments(self.segments_lists[buffer_time], self.merge_tolerances[buffer_time]) # 使用 merge_tolerances
                    # 生成翻译文本
                    self.show_texts[buffer_time] = self.generate_translation(merged_segments,buffer_time)
            global_outputs.append(self.show_texts[buffer_time])

        combined_text = self.get_combined_text()
        # outputs.append(combined_text)
        global_outputs.insert(0, combined_text)  # 将 combined_text 插入 outputs 的第一个元素

        # if stream is not None:
        #     stream = np.concatenate([stream, y])
        # else:
        #     stream = y
        return stream, *global_outputs  # 将所有文本输出


# 创建 Transcriber 实例和 Gradio 界面
transcriber = Transcriber()

# punctuator = Punctuator()
# test_text = """
# that's why i rely on grammar-ly-go to help bring out my full-potential as a content creator
# i know first how important sometimes is to hook your audience with great ideas and compelling content I am a youtuber 
# he likes a reminder and inspires you to always keep learning
# Good--NB Nike_ct
# the next person i embrace is which is known for their nobleness or compassion it might seem a bit unexpected to associate samurai's martial 
# yesterday . by adopting thiset , i hope that even the most intimidating tasks become approachable and the journey to mastery evolves into an encouraging experiment . 
# this is `ddd
# """
# g = punctuator.add_punctuation(test_text)

# print(g)

demo = gr.Interface(
    transcriber.transcribe,
    ["state", 
     gr.Audio(sources=["microphone"], streaming=True, label="点开始录音就实时更新了"),
     gr.Slider(minimum=0, maximum=9999, value=1, step=1, label="Password (0=不允许, 1024=允许) 别试密码，两个同时录音会炸掉")],
    ["state", gr.Textbox("PolyU",label="综合转录"), gr.Textbox("Hello World! Yuan", label="3秒转录"), gr.Textbox("Choose your life!", label="30秒转录"), gr.Textbox("COMP", label="60秒转录")],
    live=True,
    title="霸道翻译爱上我：字字珠玑，话话情深，沅式翻译之我叫翻译机",
    description="直接点开始录音就行了，有延迟，脚本会自动更新，只是提供辅助，保证的是准确率，3秒转录的可能不是那么准"
)

transcriber.clear_temp_dir()
demo.launch(share=True, show_error=True)
