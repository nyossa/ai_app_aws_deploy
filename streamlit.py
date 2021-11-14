import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
import pickle
from sklearn.preprocessing import MinMaxScaler
import japanize_matplotlib #matplotlibのラベルの文字化け解消のためインストール
import datetime
import matplotlib.ticker as mtick #グラフ描画時にy軸に%表示する。
from model import LSTM_Corona
from PIL import Image
import io 
import torch #BERT動かすにはtorchライブラリが必要。
from torch import nn #ソフトマックス関数使用。
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from transformers import BertForSequenceClassification, BertJapaneseTokenizer


# モデルのパス
LSTM_MODEL_FILE_PATH = './model/lstm/lstm.pickle'
BERT_MODEL_DIR_PATH = './model/bert' #モデル関連ディレクトリ
BERT_MODEL_FILE_PATH =  './model/bert/pytorch_model.bin' #BERTモデル本体
RESNET_MODEL_FILE_PATH =  './model/resnet/tl_resnet50_cpu.pth' #RESNETモデル本体
#データのパス
CIFAR100_PATH = './data/CIFAR100'
covid19_data = './data/time_series_covid19_confirmed_global.csv'

#基準年月日
base_y = 2021
base_m = 11
base_d = 14

window_size = 30

#データの中で0で変化がないところを削る。
df = pd.read_csv(covid19_data)
df = df.iloc[:,37:]

#日ベースごとに全世界を足して、各日ベースの世界全体の感染者数求める
daily_global = df.sum(axis=0)

#日付の値をpandasのdatetimeに変換
daily_global.index = pd.to_datetime(daily_global.index)

y = daily_global.values.astype(float)

def main():
    #タイトルの表示
    st.title('解析処理')

    selected_item = st.selectbox('・解析処理を選択して下さい。',
                                 ['', '画像分類（RESNET）', 'Covid19予測（LSTM）', '文章分類（BERT）'])
    
    if selected_item == '画像分類（RESNET）':
        st.write('CIFAR-100の100クラスに画像を分類します。')
        uploaded_file = st.file_uploader('分類したい画像をアップロードして下さい。')

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            st.image(
                image, caption='upload images',
                use_column_width=True #画像の横幅をカラム幅に合わせす。
            )

            out = analyze_resnet(image)
            out_F = F.softmax(out, dim=1)

            out_F, batch_indices = out_F.sort(dim=1, descending=True)

            # cifar100のクラス名取得
            class_name_cifar100 = get_cifar100_classes()

            # 描画数
            plot_size = 10

            class_list = [] #クラス名格納用リスト
            predict_list = np.empty((0, plot_size), np.float32)#予測値格納用空のnumpy配列

            predict = out_F.detach().numpy() 

            for probs, indices in zip(predict, batch_indices):
                for k in range(plot_size):
                    #st.write(f"Top-{k + 1} {class_name_cifar100[indices[k]]} {probs[k]:.2%}")
                    class_list.append(class_name_cifar100[indices[k]]) #python配列へ追加
                    predict_list = np.append(predict_list,probs[k])    #numpy配列へ追加

            #解析結果の描画
            fig, ax = plt.subplots()
            x = np.arange(len(class_list)) 
            y = np.round(predict_list*100).astype(int)#結果を%表示するので四捨五入しint型に変換。
            plt.title("解析結果")
            plt.xlabel("クラス", fontsize=13)
            plt.ylabel("確率", fontsize=13)
            plt.grid(linestyle='dotted', linewidth=1)
            plt.bar(x, y, label='カテゴリー', align='center', alpha=0.7)
            plt.xticks(x, class_list, fontsize=8, rotation=45)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())#y軸を%表示
            st.pyplot(fig)

    elif selected_item == 'Covid19予測（LSTM）':
        selected_item = st.selectbox('・2021年11月14日以降の感染者数を予測します。何日後まで予測するか選択して下さい。',
                                 ['', '10日後', '20日後', '30日後', '60日後'])
        #予測ボタン
        start = st.button('予測開始')

        if start and not selected_item:
            st.write('<span style="color:red;">予測する期間を選択して下さい。</span>', unsafe_allow_html=True)

        if start and selected_item:
            date_dict = {'10日後':10, '20日後':20, '30日後':30, '60日後':60}
            sel_datte = date_dict[selected_item]

            d = datetime.date(base_y, base_m, base_d)
            target_d = d + datetime.timedelta(days=sel_datte)
            base_date = str(base_y) + '-' + str(base_m) + '-' + str(base_d) #基準日

            out = analyze_lstm(sel_datte)

            #講座では2020年データ使用しており3日間しか予測しないので、グラフ化した時に見にくいので、１ヶ月分を予測するようにする。
            x = np.arange(base_date,target_d, dtype='datetime64[D]').astype('datetime64[D]')

            fig, ax = plt.subplots(figsize=(12,5))
            plt.title('コロナウィルスの全世界感染者数')
            plt.ylabel("感染者数")
            plt.xlabel("日付")
            plt.grid(linestyle='dotted', linewidth=1)
            plt.gca().ticklabel_format(style='plain', axis='y')#y軸を省略せずにメモリ表示
            plt.plot(daily_global, label='確定値')#オリジナルデータ
            plt.plot(x, out[window_size:],label='予測結果')#予測値
            plt.legend(loc='best')
            st.pyplot(fig)

    elif selected_item == '文章分類（BERT）':
        text = st.text_area('・カテゴリー分類する記事を入力して下さい。')
        
        st.session_state.start = False

        #解析ボタン
        start = st.button('解析開始')
        
        if start:
            st.session_state.start = True

        #if start and not text:
        if st.session_state.start and not text:
            st.write('<span style="color:red;">解析する記事を入力して下さい。</span>', unsafe_allow_html=True)
       
        if st.session_state.start and text:
            
            #カテゴリー辞書
            category_dict = {'dokujo-tsushin':'独女通信',
                            'livedoor-homme':'livedoor HOMME',
                            'kaden-channel':'家電チャンネル',
                            'smax':'エスマックス',
                            'topic-news':'トピックニュース',
                            'peachy':'Peachy',
                            'movie-enter':'MOVIE ENTER',
                            'it-life-hack':'ITライフハック',
                            'sports-watch':'Sports Watch'}
            
            #カテゴリー辞書のキーリスト
            category_key_list = list(category_dict.keys())
            #カテゴリー辞書の値リスト
            category_values_list = list(category_dict.values())

            #改行\n、タブ\t、復帰\r、全角スペース\u3000を除去
            text = [sentence.strip() for sentence in text]
            text = list(filter(lambda line: line != '', text))
            text = ''.join(text)
            text = text.translate(str.maketrans({"\n":"", "\t":"", "\r":"", "\u3000":""})) 
            #解析実行
            out = analyze_bert(text)
            #出力結果が確率ではないためソフトマックスに通して確率にする。
            out_F = F.softmax(out[0], dim=1)

            #Tensor型からnumpyに変換→detach()関数でデータ部分を切り離し、numpy()でnumpyに変換する。
            predict = out_F.detach().numpy() 

            #解析結果の描画
            fig, ax = plt.subplots()
            x = np.arange(len(predict[0])) 
            y = np.round(predict[0]*100).astype(int)#結果を%表示するので四捨五入しint型に変換。

            plt.title("解析結果")
            plt.xlabel("カテゴリー", fontsize=13)
            plt.ylabel("確率", fontsize=13)
            plt.grid(linestyle='dotted', linewidth=1)
            plt.bar(x, y,  label='カテゴリー', align='center', alpha=0.7)
            plt.xticks(x, category_values_list, fontsize=8, rotation=45)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())#y軸を%表示
            st.pyplot(fig)

#Resnet解析
def analyze_resnet(img):

    model, transforms = load_resnet_model()
    inputs = transforms(img)
    inputs = inputs.unsqueeze(0) #unsqueezeは元のテンソルを書き換えずに、次元を増やしたテンソルを返す。

    out = model(inputs)

    return out

#LSTM解析
def analyze_lstm(future=10):
    #モデルの読み込み
    with open(LSTM_MODEL_FILE_PATH, mode="rb") as f:
        model = pickle.load(f)
    
    #入力のデータを正規化（-1〜0に収まるように変換）
    scaler = MinMaxScaler(feature_range=(-1,1))

    y_normalized = scaler.fit_transform(y.reshape(-1,1))
    y_normalized = torch.FloatTensor(y_normalized).view(-1)
    preds = y_normalized[-window_size:].tolist()

    model.eval()#評価モード

    for i in range(future):
        sequence = torch.FloatTensor(preds[-window_size:])
    
        with torch.no_grad():#勾配の計算の無効化
            model.hidden =(torch.zeros(1,1,model.h_size),torch.zeros(1,1,model.h_size))#隠れ層NO無効化
            preds.append(model(sequence).item())#その都度計算された予測値を格納
        
    #予測値が正規化されてるので元のスケールに戻す。
    out = scaler.inverse_transform(np.array(preds).reshape(-1,1))

    return out

#BERT解析
def analyze_bert(text):

     loaded_model, loaded_tokenizer = load_bert_model()
     max_length = 512
     words = loaded_tokenizer.tokenize(text)
     word_ids = loaded_tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
     word_tensor = torch.tensor([word_ids[:max_length]])  # テンソルに変換
     out = loaded_model(word_tensor)
    
     return out

# Resnetのパラメータ読み込み
@st.cache(allow_output_mutation=True)
def load_resnet_model():
    # Resnetモデル取得
    model = torchvision.models.resnet50(pretrained=True)
    # モデルの最終層を100個のクラスの予測用に改良する。
    model.fc = nn.Linear(2048, 100)
    # モデルを評価モードにする
    model.eval()
    model.load_state_dict(torch.load(RESNET_MODEL_FILE_PATH))

    # transform取得
    transform = transforms.Compose(
        [
            transforms.Resize(256),  # (256, 256) で切り抜く。
            transforms.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
            transforms.ToTensor(),  # テンソルにする。
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 標準化する。
        ]
    )

    return model, transform

# CIFARのクラス名取得
@st.cache(allow_output_mutation=True)
def get_cifar100_classes():
    trainset = torchvision.datasets.CIFAR100(root=CIFAR100_PATH,download=True)
    return trainset.classes

@st.cache(allow_output_mutation=True)
def load_bert_model():
    # モデル取得
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR_PATH)

    # tokenizer取得
    tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL_DIR_PATH)

    return model, tokenizer

if __name__ == "__main__":
    main()
