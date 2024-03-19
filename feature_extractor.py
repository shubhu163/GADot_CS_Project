import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split


def reshape_samples(samples, num_packets=10, num_features=11):
    reshaped_samples = []
    for sample in samples:
        if len(sample) > num_packets:
            sample = sample[:num_packets]
        elif len(sample) < num_packets:
            padding = np.zeros((num_packets - len(sample), num_features))
            sample = np.vstack((sample, padding))
        reshaped_samples.append(sample)
    return np.array(reshaped_samples)

X_reshaped = reshape_samples(X_scaled)


dataset_path = '/nfs/hpc/share/joshishu/CS_Project/GADot_CS_Project/data/CICD_DATA/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(dataset_path)

features = ['Flow Duration', 'Total Length of Fwd Packets','Total Length of Bwd Packets', 'Highest Layer', 'IP Flags', 'Protocols',
            'TCP Len', ' ACK Flag Count', 'TCP Flags', 'TCP Win Size', 'UDP Len', 'ICMP Type']
X = df[features]
y = df['Label']  


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


def reshape_samples(samples, num_packets=10, num_features=11):
    reshaped_samples = []
    for sample in samples:
        if len(sample) > num_packets:
            sample = sample[:num_packets]
        elif len(sample) < num_packets:
            padding = np.zeros((num_packets - len(sample), num_features))
            sample = np.vstack((sample, padding))
        reshaped_samples.append(sample)
    return np.array(reshaped_samples)

X_reshaped = reshape_samples(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)


