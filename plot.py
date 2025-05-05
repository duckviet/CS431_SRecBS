import re
import matplotlib.pyplot as plt

def read_log_file(filename):
    epochs = []
    val_ndcg = []
    val_hr = []
    test_ndcg = []
    test_hr = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(
                r"(\d+)\s+\(([\d\.]+),\s*([\d\.]+)\)\s+\(([\d\.]+),\s*([\d\.]+)\)", line
            )
            if match:
                epochs.append(int(match.group(1)))
                val_ndcg.append(float(match.group(2)))
                val_hr.append(float(match.group(3)))
                test_ndcg.append(float(match.group(4)))
                test_hr.append(float(match.group(5)))
    return epochs, val_ndcg, val_hr, test_ndcg, test_hr

# Tên file và tên model
files = [
    ('ml-1m_default/SASRec_log.txt', 'SASRec'),
    ('ml-1m_default/SASRec_TCE_log.txt', 'SASRec_TCE'),
    ('ml-1m_default/S3Rec_SMA_log.txt', 'S3Rec_SMA'),
    ('ml-1m_default/S3Rec_TCE_log.txt', 'S3Rec_TCE')
]

# Đọc dữ liệu
data = [read_log_file(f[0]) for f in files]

# Vẽ 2 biểu đồ: NDCG và HR
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Màu cho từng model
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# NDCG
for i, (epochs, val_ndcg, val_hr, test_ndcg, test_hr) in enumerate(data):
    model_name = files[i][1]
    axes[0].plot(
        epochs, val_ndcg, label=f'{model_name} Val', color=colors[i], linestyle='-'
    )
    axes[0].plot(
        epochs, test_ndcg, label=f'{model_name} Test', color=colors[i], linestyle='--'
    )
axes[0].set_title('NDCG')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Score')
axes[0].grid(True)
axes[0].legend()

# HR
for i, (epochs, val_ndcg, val_hr, test_ndcg, test_hr) in enumerate(data):
    model_name = files[i][1]
    axes[1].plot(
        epochs, val_hr, label=f'{model_name} Val', color=colors[i], linestyle='-'
    )
    axes[1].plot(
        epochs, test_hr, label=f'{model_name} Test', color=colors[i], linestyle='--'
    )
axes[1].set_title('HR')
axes[1].set_xlabel('Epoch')
axes[1].grid(True)
axes[1].legend()

fig.suptitle('So sánh NDCG và HR của 3 model (Val & Test)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Lưu ảnh
plt.savefig('compare_models_ndcg_hr.png', dpi=300)

plt.show()
