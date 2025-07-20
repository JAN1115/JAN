import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from sklearn.model_selection import train_test_split

# === 파라미터 ===
SEQUENCE_LENGTH = 10
MODEL_FILENAME = 'baccarat_lstm_advanced_multi.h5'
CSV_FILENAME = 'baccarat_data.csv'

# === 데이터 로딩 및 인코딩 ===
df = pd.read_csv(CSV_FILENAME)
data = [0 if r == 'P' else 1 for r in df['results'] if r in ['P', 'B']]

def extract_features(seq):
    # 각 시점: [0/1, 줄길이, 퐁당, 3턴 P수, 변동성]
    features = []
    for i in range(len(seq)):
        streak = 1
        for j in range(i-1, -1, -1):
            if seq[j] == seq[i]:
                streak += 1
            else:
                break
        zigzag = 1 if (i >= 1 and seq[i] != seq[i-1]) else 0
        ngram3 = sum(seq[max(0,i-2):i+1])
        volatility = sum(1 for k in range(1, i+1) if seq[k] != seq[k-1])/(i+1) if i > 0 else 0
        features.append([seq[i], streak, zigzag, ngram3, volatility])
    return features

# === 시퀀스/피처 생성 ===
sequences, labels1, labels3 = [], [], []
for i in range(len(data) - SEQUENCE_LENGTH - 3):
    seq = data[i:i+SEQUENCE_LENGTH]
    feats = extract_features(seq)
    sequences.append(feats)
    labels1.append(data[i+SEQUENCE_LENGTH])      # 다음 1턴
    labels3.append(data[i+SEQUENCE_LENGTH+2])    # 다음 3턴 후

X = np.array(sequences)
y1 = np.array(labels1)
y3 = np.array(labels3)

X_train, X_test, y1_train, y1_test, y3_train, y3_test = train_test_split(
    X, y1, y3, test_size=0.2, random_state=42
)

# === Bidirectional LSTM 멀티아웃풋 모델 정의 ===
inp = Input(shape=(SEQUENCE_LENGTH, 5))
x = Bidirectional(LSTM(64, return_sequences=True))(inp)
x = Dropout(0.3)(x)
x = LSTM(32)(x)
d = Dense(16, activation='relu')(x)
out1 = Dense(1, activation='sigmoid', name='next1')(d) # 다음 1턴
out3 = Dense(1, activation='sigmoid', name='next3')(d) # 다음 3턴

model = Model(inp, [out1, out3])
model.compile(
    optimizer='adam',
    loss={'next1': 'binary_crossentropy', 'next3': 'binary_crossentropy'},
    metrics={'next1': 'accuracy', 'next3': 'accuracy'}
)

model.summary()
print("강화 Bidirectional LSTM 모델 훈련 시작...")
history = model.fit(
    X_train, {'next1': y1_train, 'next3': y3_train},
    validation_data=(X_test, {'next1': y1_test, 'next3': y3_test}),
    epochs=8,
    batch_size=128,
    verbose=1
)

model.save(MODEL_FILENAME)
print(f"\n✅ Bidirectional 멀티아웃풋 모델 훈련 완료! '{MODEL_FILENAME}' 파일이 저장되었습니다.")
