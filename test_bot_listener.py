'''
bot_listener.py

ウェイクワードを認識してサウンドを鳴らし、コマンド入力を待機する音声認識プログラムです。
ウェイクワードが入力されるとコマンド受付モードに入り、終了コマンドが入力されると待機モードに戻ります。
音声認識結果を返し、ChatGPTにデータを渡します。
'''

import json, time
from pathlib import Path

import queue
import sys
from collections import namedtuple

import sounddevice as sd

from vosk import Model, KaldiRecognizer, SetLogLevel


# from bot_motor_controller import neopixels_face, neopixels_hearing, neopixels_off
from test_bot_voice_synthesizer import notification


# Jsonファイルからウェイクワードとコマンドの配列を読み込む
with open(Path("data/command_data.json"), "rb") as f:
    data = json.load(f)

WAKE = data["wake"]
EXIT = data["exit"]

class MicrophoneStream:
    """マイク音声入力のためのクラス."""

    def __init__(self, rate, chunk):
        """音声入力ストリームを初期化する.

        Args:
           rate (int): サンプリングレート (Hz)
           chunk (int): 音声データを受け取る単位（サンプル数）
        """
        # マイク入力のパラメータ
        self.rate = rate
        self.chunk = chunk

        # 入力された音声データを保持するデータキュー（バッファ）
        self.buff = queue.Queue()

        # マイク音声入力の初期化
        self.input_stream = None

    def open_stream(self):
        """マイク音声入力の開始"""
        self.input_stream = sd.RawInputStream(
            samplerate=self.rate,
            blocksize=self.chunk,
            dtype="int16",
            channels=1,
            callback=self.callback,
        )

    def callback(self, indata, frames, time, status):
        """音声入力の度に呼び出される関数."""
        if status:
            print(status, file=sys.stderr)

        # 入力された音声データをキューへ保存
        self.buff.put(bytes(indata))

    def generator(self):
        """音声認識に必要な音声データを取得するための関数."""
        while True:  # キューに保存されているデータを全て取り出す
            # 先頭のデータを取得
            chunk = self.buff.get()
            if chunk is None:
                return
            data = [chunk]

            # まだキューにデータが残っていれば全て取得する
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            # yieldにすることでキューのデータを随時取得できるようにする
            yield b"".join(data)

SetLogLevel(-1)  # VOSK起動時のログ表示を抑制

# 入力デバイス情報に基づき、サンプリング周波数の情報を取得
# sd.default.device = [2, 3]
input_device_info = sd.query_devices(kind="input")
# print(input_device_info)
sample_rate = int(input_device_info["default_samplerate"])
# print(sample_rate)
sample_rate = 16000 # 44100
chunk_size = 8000 # 8820

# マイク入力を初期化
mic_stream = MicrophoneStream(sample_rate, chunk_size)

# Voskモデルの読み込み
# model = Model(str(Path("vosk-model-small-ja-0.22").resolve()))
# model = Model(str(Path("vosk-model-ja-0.22").resolve()))
model = Model(lang="ja")

# 音声認識器を構築
recognizer = KaldiRecognizer(model, sample_rate)

# マイク入力ストリームおよび音声認識器をまとめて保持
VoskStreamingASR = namedtuple(
    "VoskStreamingASR", ["microphone_stream", "recognizer"]
)
vosk_asr = VoskStreamingASR(mic_stream, recognizer)

def get_asr_result(vosk_asr):
    """音声認識APIを実行して最終的な認識結果を得る.

    Args:
       vosk_asr (VoskStreamingASR): 音声認識モジュール

    Returns:
       recog_text (str): 音声認識結果
    """
    mic_stream = vosk_asr.microphone_stream
    mic_stream.open_stream()
    with mic_stream.input_stream:
        audio_generator = mic_stream.generator()
        for content in audio_generator:
            if vosk_asr.recognizer.AcceptWaveform(content):
                recog_result = json.loads(vosk_asr.recognizer.Result())
                recog_text = recog_result["text"].split()
                recog_text = "".join(recog_text)  # 空白記号を除去
                return recog_text
        return None


# ウェイクワード待機をlistening コマンド待機をhearingと設定
listening = True
hearing = False

# listeningをループして音声認識 ウェイクワード認識でhearingループする
def bot_listen_hear():
    global listening, hearing
    
    # neopixelsの目を点灯
    # neopixels_face()
    if hearing == True: print("🖥️ SYSTEM: ","-"*22, "GPTに話しかけてください","-"*22)
    else: print("🖥️ SYSTEM: ","-"*22, "ウェイクワード待機中","-"*22)
    
    while listening:
        response = get_asr_result(vosk_asr) # engine()
        print(response)
        if response in WAKE:
            listening = False
            hearing = True
            # neopixels_off()
            notification()
            time.sleep(0.5)
            # neopixels_hearing()
            print("🖥️ SYSTEM: ","-"*22, "GPTに話しかけてください","-"*22)
        elif response.strip() == "":
            continue  # 空白の場合はループを続ける
        else:
            pass
    
    while hearing:
        response = get_asr_result(vosk_asr) # engine()
        print(response)
        if response in EXIT:
            listening = True
            hearing = False
            # neopixels_off()
            notification()
            time.sleep(0.5)
            # neopixels_hearing()
        elif response.strip() == "":
            continue  # 空白の場合はループを続ける
        else:
            # neopixels_off()
            notification()
            time.sleep(0.5)
            # neopixels_hearing()
        return response 

if __name__ == "__main__":
    while True:
        response = bot_listen_hear()
        print("response: ",response)
