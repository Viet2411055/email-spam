import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# LOAD DATA
df = pd.read_csv('spam_ham_dataset.csv')

df = df[['text', 'label_num']]
df.columns = ['content', 'label']

# CLEAN TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[\r\n]+', ' ', text)
    return text.strip()

df['content'] = df['content'].apply(clean_text)

# VECTORIZATION
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['content'])
y = df['label'].values

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODELS
models = {
    "Logistic": LogisticRegression(max_iter=200, solver='lbfgs'),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

# TRAIN + EVALUATE
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("="*40)
    print(name)
    print(classification_report(y_test, y_pred))

    # ==============================
# PLOT COST FUNCTION (Log-loss)
# ==============================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(y, h):
    m = len(y)
    return -(1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

# Giả lập cost giảm theo iteration
iterations = 50
costs = []

# lấy 1 phần nhỏ dữ liệu để minh họa
y_sample = y_train[:100]

for i in range(1, iterations+1):
    # giả lập z thay đổi dần
    z = np.linspace(-2 + i*0.05, 2 + i*0.05, len(y_sample))
    h = sigmoid(z)
    
    cost = compute_cost(y_sample, h)
    costs.append(cost)

# ==============================
# PLOT SIGMOID
# ==============================
z_vals = np.linspace(-10, 10, 100)
sig_vals = sigmoid(z_vals)

# ==============================
# DRAW FIGURE
# ==============================
plt.figure(figsize=(12,5))

# Cost plot
plt.subplot(1,2,1)
plt.plot(range(iterations), costs)
plt.title("Cost Function (Log-Loss)")
plt.xlabel("Iterations")
plt.ylabel("Cost")

# Sigmoid plot
plt.subplot(1,2,2)
plt.plot(z_vals, sig_vals, color='red')
plt.axvline(0, linestyle='--')
plt.axhline(0.5, linestyle='--')
plt.title("Sigmoid g(z)")
plt.xlabel("z = theta^T * x")
plt.ylabel("Probability h(x)")

plt.tight_layout()
plt.show()