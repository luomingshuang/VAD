# encoding:utf-8
import collections
import contextlib
import wave
import base64
import urllib
import requests
import webrtcvad
import struct
import json
import multiprocessing
import argparse
import os
import logging
import re
import chardet
#import sys
#import time
#import tqdm
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='app_trans_local_wav_offline.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s [line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self,
                 bytes,
                 timestamp,
                 duration
                 ):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class Wave:
    """define a wave class"""

    def __init__(self,
                 wav_path,
                 wav_name
                 ):
        assert(wav_name.endswith("wav"))

        self.is_8k = 0
        self.is_16k = 0
        self.is_pcm = 0
        self.not_pcm = 0
        self.big_than_16k = 0

        self.wav_path = wav_path
        self.wav_name = wav_name
        self.wav_full_path = wav_path + wav_name
        self.fp = open(self.wav_full_path, "rb").read()
        self.AudioFormat = struct.unpack('<H', self.fp[20:22])[0]
        self.SampleRate = struct.unpack('<L', self.fp[24:28])[0]
        self.header = self.choose_header()
        self.check_format()

        self.ChunkSize = struct.unpack('<L', self.fp[4:8])[0]
        self.Format = self.fp[8:12]
        self.Subchunk1Size = struct.unpack('<L', self.fp[16:20])[0]
        self.NumChannels = struct.unpack('<H', self.fp[22:24])[0]
        self.SampleRate = struct.unpack('<L', self.fp[24:28])[0]
        self.ByteRate = struct.unpack('<L', self.fp[28:32])[0]
        self.BlockAlign = struct.unpack('<H', self.fp[32:34])[0]
        self.BitsPerSample = struct.unpack('<H', self.fp[34:36])[0]
        # self.NumOfBytes = struct.unpack('<I', self.fp[40:44])[0]
        self.data = self.fp[44:]
        self.NumOfBytes = len(self.data)
        self.NumOfSamples = self.NumOfBytes / self.BitsPerSample * 8.0 / self.NumChannels
        self.WavLenSeconds = self.NumOfSamples / 1.0 / self.SampleRate
    
    def check_format(self):
        if self.AudioFormat == 1:
            self.is_pcm = 1
            logging.info("[check format]" + self.wav_name + "[ok]")
            pass
        else:
            self.not_pcm = 1
            logging.info("[check format]" + self.wav_name + "[ox]")
            logging.info("[start convert format]" + self.wav_name)
            temp_path = "./tmp"
            temp_wav = temp_path + "/" + self.wav_name
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
            cmd = "ffmpeg  -i " + self.wav_full_path + " -loglevel quiet -acodec pcm_s16le " + temp_wav +" -y"
            os.system(cmd)
            logging.info("[Finish convert format]" + self.wav_name)
            self.wav_full_path = temp_wav
            self.fp = open(self.wav_full_path, "rb").read()

    def verbose(self):
        logging.info("*************************AUDIO INFORMATION******************************")
        logging.info("audio file name :" + str(self.wav_name))
        logging.info("audio format    :" + str(self.AudioFormat))
        logging.info("num of channels :" + str(self.NumChannels))
        logging.info("sample rate     :" + str(self.SampleRate))
        logging.info("bits per sample :" + str(self.BitsPerSample))
        logging.info("chunk size      :" + str(self.ChunkSize))
        logging.info("Format          :" + str(self.Format))
        logging.info("Num of Bytes    :" + str(self.NumOfBytes))
        logging.info("Num of Samples  :" + str(self.NumOfSamples))
        logging.info("WavLen  seconds :" + str(self.WavLenSeconds))
        logging.info("************************************************************************")

    def choose_header(self):
        header_16k = open("../resource/header_16k.wav", "rb").read()[:36]+b'data'
        header_8k = open("../resource/header_8k.wav", "rb").read()[:36]+b'data'
        if self.SampleRate == 8000:
            self.is_8k =1
            return header_8k
        elif self.SampleRate == 16000:
            self.is_16k =1
            return header_16k
        else:
            self.big_than_16k =1
            logging.info("Sample Rate should be 8000 or 16000" + str(self.wav_name))
            return

class MixSpeech(Wave):
    def __init__(self,
                 wav_dir,
                 wav_file,
                ):
        Wave.__init__(self,wav_dir,wav_file) 
        self.url_8k = '' #ivr test
        self.url_16k = '' #ivr test
        self.wav_data = open(wav_dir+wav_file,'rb').read()
    def mixspeech(self):
        bytesStr = base64.b64encode(self.wav_data)
        #print('hi')
        d = {"wav_str": bytesStr, 'output_mode_str': 'hanzi_pinyin','hotwords_str':''}
        e = urllib.parse.urlencode(d)
        if self.SampleRate == 8000:
            n = requests.post(self.url_8k, data=e,
                              headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
        elif self.SampleRate == 16000:
            n = requests.post(self.url_16k, data=e,
                              headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
        else:
            logging.info("Samplerate not supported" + str(self.SampleRate))
            return
        tmp = json.loads(n.text)
        tmp2 = eval(tmp)
        return tmp2

class VadAsr(Wave):
    def __init__(self,
                 wav_dir,
                 json_dir,
                 wav_file,
                 vad_num=1
                 ):
        Wave.__init__(self,wav_dir,wav_file) 
        self.json_dir = json_dir
        self.url_8k = '' #ivr test
        self.url_16k = '' #ivr test
        self.vad = webrtcvad.Vad(vad_num)
        self.json_list = []
        self.jsonfp = open(self.json_dir + self.wav_name.split('.')[0] + ".json", 'w')
        self.num_dict = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
        self.nums = "0123456789"
        self.p = r"[0-9]{3,}"
        self.pattern = re.compile(self.p) 

    def to_unicode(self, unicode_or_str):
        '''
        Convert to unicode encoding.
        :param unicode_or_str:Input string or unicode content.
        :return: content with unicode format.
        '''
        if isinstance(unicode_or_str, str):
            value = unicode_or_str.decode('utf-8')
        else:
            value = unicode_or_str
        return value

    def to_str(self, unicode_of_str):
        '''
        Convert to UTF-8 encoding string.
        :param unicode_of_str:
        :return:
        '''
        if isinstance(unicode_of_str, unicode):
            value = unicode_of_str.encode('utf-8')
        else:
            value = unicode_of_str
        return value

    def num_sub(self, ori):
        return '$'*len(str(ori.group(0)))

    def num_transcript(self, ori):
        ori = self.to_str(ori)
        temp = ori
        temp = re.sub(self.pattern, self.num_sub, temp)
        res = ""
        for i in range(len(temp)):
            if((ori[i] in self.nums) and (temp[i] != '$')):
                res += self.num_dict[int(ori[i])]
            else:
                res += ori[i]
        res = res.replace('|','，')
        res = res + '。'
        res = self.to_unicode(res)
        return res

    def mixspeech(self, chunk_file_name):
        #print(type(chunk_file_name))
        #print(len(chunk_file_name))
        bytesStr = base64.b64encode(chunk_file_name)
        #print('bs',bytesStr[:10])
        d = {"wav_str": bytesStr, 'output_mode_str': 'hanzi_pinyin','hotwords_str':''}
        e = urllib.parse.urlencode(d)
        if self.SampleRate == 8000:
            n = requests.post(self.url_8k, data=e,
                              headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
        elif self.SampleRate == 16000:
            n = requests.post(self.url_16k, data=e,
                              headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
        else:
            logging.info("Samplerate not supported" + str(self.SampleRate))
            return
        tmp = json.loads(n.text)
        tmp2 = eval(tmp)
        #print(tmp2)
        return tmp2

    def read_wave(self, path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def write_wave(self, path, audio, sample_rate):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms,
                      padding_duration_ms, vad, frames):
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.3 * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.3 * ring_buffer.maxlen:
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames]), voiced_frames[0].timestamp, voiced_frames[
                        -1].timestamp + voiced_frames[-1].duration
                    ring_buffer.clear()
                    voiced_frames = []
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames]), voiced_frames[0].timestamp, voiced_frames[-1].timestamp + \
                  voiced_frames[-1].duration

    def split2mono(self, fp):
        #print(type(fp))
        #print(fp[:10])
        audio_operator = list(fp[44:])
        audio_client = list(fp[44:])
        del audio_client[2::4]
        del audio_client[2::3]
        del audio_operator[0::4]
        del audio_operator[0::3]
        #print(audio_operator[:10])
        audio_operator= bytes(audio_operator)
        #print(type(audio_operator))
        #print(audio_operator[:10])
        #audio_operator = b''.join(audio_operator)
        #audio_operator = ''.join(str(i) for i in audio_operator).encode()
        #audio_client = b''.join(audio_client)
        #audio_client = ''.join(str(i) for i in audio_client).encode()
        audio_client= bytes(audio_client)
        return audio_client, audio_operator

    def write2json(self, audio_input, sample_rate, role):
        from tqdm import tqdm
        frames = self.frame_generator(30, audio_input, sample_rate)
        frames = list(frames)
        segments = self.vad_collector(sample_rate, 30, 300, self.vad, frames)
        segments=list(segments)
        for segment in range(len(segments)):
            print(segment,segments[segment][2]-segments[segment][1])
        #print(len((segments)))
        #len_total=len((segments))
        index = 0
        #for i, segment in enumerate(segments):
        for segment in tqdm(segments):
            datasize = len(segment[0])
            datasize_byte = struct.pack('<L', datasize)
            seg_wav = self.header + datasize_byte + segment[0]
            #wf=open('index_'+str(i)+'.wav','wb')
            #wf.write(seg_wav)
            #wf.close()
            trans_tmp = self.mixspeech(seg_wav)
            trans = trans_tmp['response']
            score = trans_tmp['score']
            if trans:
                json_line = json.dumps({'index': index, 'text': trans,'score':score, 'sex': "", 'role': role, 'xmax': segment[2],
                                        'xmin': segment[1], 'correct': 1}, ensure_ascii=False)
                self.json_list.append(json_line)
                index += 1
                #sys.stdout.write("line "+str(index))
                #sys.stdout.write("\r")
                #sys.stdout.flush()
        #sys.stdout.write("\n"+"Finish")

    def trans_dual(self):

        audio_client, audio_operator = self.split2mono(self.fp)

        """
        write client result into json
        """
        self.write2json(audio_client, self.SampleRate, 'client')
        """
        operator
        """
        self.write2json(audio_operator, self.SampleRate, 'operator')

        new_json_list = []
        for line in self.json_list:
            new_json_list.append(json.loads(line))
        new_combine_list = sorted(new_json_list, key=lambda k: int(1000.0 * float(k['xmin'])))
        my_index = 0
        for line in new_combine_list:
            line['index'] = my_index
            my_index = my_index + 1
            save_line = json.dumps(line, ensure_ascii=False)
            self.jsonfp.write(save_line + '\n')

    def trans_mono(self):

        self.write2json(self.data, self.SampleRate, 'None')
        new_json_list = []
        for line in self.json_list:
            new_json_list.append(json.loads(line))
        new_combine_list = sorted(new_json_list, key=lambda k: int(1000.0 * float(k['xmin'])))
        my_index = 0
        for line in new_combine_list:
            line['index'] = my_index
            my_index = my_index + 1
            save_line = json.dumps(line, ensure_ascii=False)
            self.jsonfp.write(save_line + '\n')

    def trans_any(self):
        if self.NumChannels == 1:
            logging.info("[start mono channel audio trans]" + str(self.wav_name))
            self.trans_mono()
        elif self.NumChannels == 2:
            logging.info("[start stero audio trans]" + str(self.wav_name))
            self.trans_dual()
        else:
            logging.info("[num of channels should be equal to 1 or 2 ]" + str(self.wav_name))
            return


def trans_wav_list(wav_dir,
                   wav_list,
                   json_dir,
                   thread_index,
                   ):
    num_wav = len(wav_list)
    index = 1
    totalLen = 0.0
    num_all = 0
    num_not_wav = 0
    num_8k_ok = 0
    num_16k_ok = 0
    num_pcm = 0
    num_not_pcm = 0
    num_big_than_16k = 0
    for wav in wav_list:
        num_all += 1
        if not wav.endswith("wav"):
            num_not_wav += 1
            logging.warning("[it is not wav file , skipped]" + str(wav))
        else:
            mytrans = VadAsr(wav_dir=wav_dir, json_dir=json_dir, wav_file=wav, vad_num=3)
            totalLen += mytrans.WavLenSeconds

            num_8k_ok += mytrans.is_8k
            num_16k_ok += mytrans.is_16k
            num_pcm += mytrans.is_pcm 
            num_not_pcm += mytrans.not_pcm 
            num_big_than_16k += mytrans.big_than_16k

            mytrans.verbose()
            mytrans.trans_any()
            logging.warning("[process_index]: " + str(thread_index) + "[finished]: " + str(index) + "[num of all]: " + str(num_wav))
            index += 1
    return [totalLen, num_8k_ok, num_16k_ok, num_pcm, num_not_pcm, num_big_than_16k, num_not_wav, num_all]


class AsrWaves:
    def __init__(self, wav_dir, wav_list, process_num, json_dir, vad_num=3):
        self.pool = multiprocessing.Pool(processes=process_num)
        self.wav_dir = wav_dir
        self.json_dir = json_dir
        self.wav_list = wav_list
        self.len_wav_list = len(self.wav_list)
        #print(self.wav_list)
        self.process_num = process_num
        self.wav_bucket_list = []
        self.threads = []
        self.vad_num = vad_num
        self.results = [] 
        for i in range(self.process_num):
            self.wav_bucket_list.append([])

    def trans_wav_list(self,
                       wav_dir,
                       wav_list,
                       json_dir,
                       thread_index,
                       ):
        num_wav = len(wav_list)
        index = 0
        for wav in wav_list:
            mytrans = VadAsr(wav_dir=wav_dir, json_dir=json_dir, wav_file=wav, vad_num=self.vad_num)
            mytrans.verbose()
            mytrans.trans_any()
            print("process_index", thread_index, "finished ", index, "num of all", num_wav)
            index += 1

    def trans_wav_multi_process(self):
        for i in range(self.len_wav_list):
            self.wav_bucket_list[i % self.process_num].append(self.wav_list[i])

        for thread_index in range(self.process_num):
            result = self.pool.apply_async(trans_wav_list,
                                        args=(self.wav_dir,
                                              self.wav_bucket_list[thread_index],
                                              self.json_dir,
                                              thread_index,
                                              )
                                           )
            self.results.append(result)
        self.pool.close()
        self.pool.join()
        totallen = 0.0
        num_8k_ok = 0
        num_16k_ok = 0
        num_pcm = 0
        num_not_pcm = 0 
        num_big_than_16k = 0
        num_not_wav = 0
        num_all = 0
        for result in self.results:
            totallen += result.get()[0]
            num_8k_ok += result.get()[1]
            num_16k_ok += result.get()[2]
            num_pcm += result.get()[3]
            num_not_pcm += result.get()[4] 
            num_big_than_16k += result.get()[5]
            num_not_wav += result.get()[6]
            num_all += result.get()[7]
        logging.info("                   ")
        logging.info("BBBBBBBBBBBBBBBBB TASK Finished BBBBBBBBBBBBBBBBBBBBBBBBBB")
        logging.info("  total len            [" + str(totallen) + "]s Processed")
        logging.info("  num of 8k            [" + str(num_8k_ok) + "] Processed")
        logging.info("  num of 16k           [" + str(num_16k_ok) + "] Processed")
        logging.info("  num of pcm           [" + str(num_pcm) + "] Processed")
        logging.info("  num of non-pcm       [" + str(num_not_pcm) + "] Processed")
        logging.info("  num of big than 16k  [" + str(num_big_than_16k) + "] Processed")
        logging.info("  num of non-wav       [" + str(num_not_wav) + "] Processed")
        logging.info("  num of all           [" + str(num_all) + "] Processed")
        logging.info("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="trans loacl wav file into json")
    parser.add_argument("-wp",
                        "--wav_path",
                        type=str,
                        default="./data/",
                        help="input wav path"
                        )
    parser.add_argument("-jsp",
                        "--json_path",
                        type=str,
                        default="./json/",
                        help="output json path"
                        )
    parser.add_argument("-wav",
                        "--wav_name",
                        type=str,
                        default="test.wav",
                        help="single input wav name"
                        )
    parser.add_argument("-pnum",
                        "--process_num",
                        type=int,
                        default=1,
                        help="num of process used"
                        )
    parser.add_argument("-vm",
                        "--vad_num",
                        type=int,
                        default=1,
                        help="vad parameter"
                        )
    parser.add_argument("-test",
                        "--test_flag",
                        type=int,
                        default=0,
                        help="test for connection "
                        )
   
    args = parser.parse_args()
    test_flag = args.test_flag
    wav_path = args.wav_path
    json_path = args.json_path
    wav_name = args.wav_name
    process_num = args.process_num
    vad_num = args.process_num

    """
    Uasge:
    to transcribe wav in a directory 

    python vad_asr_dual.py -wp ./data/ 
                           -jsp ./json/ 
                           -pnum 6
                           -vm 3
    """
    if test_flag == 1:
        test = MixSpeech("../resource/","check_asr_service_8k.wav")
        result = test.mixspeech()
        #print(result)
        print(result['response'])
    elif True:
        wav_list = os.listdir(wav_path)
        myprocess = AsrWaves(wav_dir=wav_path,
                             process_num=process_num,
                             wav_list=wav_list,
                             json_dir=json_path,
                             vad_num=vad_num
                             )
        myprocess.trans_wav_multi_process()

    """
    Uasge:
    to transcribe wav in a directory 

    python vad_asr_dual.py -wp ./data/ 
                           -jsp ./json/ 
                           -wav test.wav 
                           -vm 3
    """
    if False:
        myasr = VadAsr(wav_dir=wav_path, wav_file=wav_name, json_dir=json_path, vad_num=vad_num)
        myasr.verbose()
        myasr.trans_any()
    
