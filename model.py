#エラーが発生するのを防ぐためにLSTM_Coronaを外部のモジュールする。
#  ・エラー内容：
#  　jupyterでLSTMモデルクラスであるLSTM_Corona作成し学習後にpickle化して出力。
#  　→streamlit.pyでpickleを読み込んで予測を行おうとしたら下記のエラーになった。
#     「AttributeError ：Can't get attribute LSTM_Corona on <module  '__main__'  from 'streamlit.py'>」
#       　→__main__モジュールにLSTM_Cornaがないと言われている。
#         →jupyter上で__main__.LSTM_Cornaというクラスをpickle化したため＝jupyter上でLSTM_Cornaというクラスを作成したら__main__.LSTM_Cornaモジュールになる。
#         →pythonコマンドで実行したファイルは、__main__モジュールとして扱われるため。
#  ・解決方法：
#      別途model.pyを作成しその中にLSTM_Coronaを記載する。そしてそれをjupyterファイルと同じ階層に置いて、jupyterファイル内でimportする。
#       →そしてjupyterファイル内ではLSTM_Coronaを定義しないことにより__main__.LSTM_Cornaモジュールではなく、model.LSTM_Cornaモジュールになる。
#       →その結果エラーが発生しなくなる。
import torch.nn as nn
import torch

class LSTM_Corona(nn.Module):
    def __init__(self, in_size=1, h_size=30, out_size=1):
        super().__init__()
        self.h_size = h_size
        self.lstm = nn.LSTM(in_size,h_size)
        self.fc = nn.Linear(h_size,out_size)
        
        self.hidden = (torch.zeros(1,1,h_size),torch.zeros(1,1,h_size))
        
    def forward(self, sequence_data):
        #lstmを実行するときは3次元のサイズを指定する必要がある。
        #１つ目の引数＝1次元目＝データのサイズ（len(sequence_data)）＝今回は30個 = train_dataNの中は30
        #2つ目の引数＝2次元目＝バッチサイズ＝今回はバッチ化していないので1
        #3つ目の引数＝3次元目＝隠れ層のサイズ＝今回なら引数で指定した30。
        lstm_out, self.hidden = self.lstm(sequence_data.view(len(sequence_data),1,-1),self.hidden)
        pred=self.fc(lstm_out.view(len(sequence_data),-1))
        
        return pred[-1]#欲しいのは最後のデータ