import pandas as pd
import numpy as np
import os
import torch
import constants as cst

def z_score_orderbook(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """DONE: remember to use the mean/std of the training set, to z-normalize the test set."""
    if (mean_size is None) or (std_size is None):
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    if (mean_prices is None) or (std_prices is None):
        mean_prices = data.iloc[:, 0::2].stack().mean()
        std_prices = data.iloc[:, 0::2].stack().std()

    price_cols = data.columns[0::2]
    size_cols = data.columns[1::2]

    for col in size_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_size) / std_size

    for col in price_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_prices) / std_prices

    if data.isnull().values.any():
        raise ValueError("data contains null value")

    return data, mean_size, mean_prices, std_size, std_prices

def normalize_messages(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None, mean_time=None, std_time=None, mean_depth=None, std_depth=None):
    if (mean_size is None) or (std_size is None):
        mean_size = data["size"].mean()
        std_size = data["size"].std()

    if (mean_prices is None) or (std_prices is None):
        mean_prices = data["price"].mean()
        std_prices = data["price"].std()

    if (mean_time is None) or (std_time is None):
        mean_time = data["time"].mean()
        std_time = data["time"].std()

    if (mean_depth is None) or (std_depth is None):
        mean_depth = data["depth"].mean()
        std_depth = data["depth"].std()

    data["time"] = (data["time"] - mean_time) / std_time
    data["size"] = (data["size"] - mean_size) / std_size
    data["price"] = (data["price"] - mean_prices) / std_prices
    data["depth"] = (data["depth"] - mean_depth) / std_depth

    if data.isnull().values.any():
        raise ValueError("data contains null value")

    data["event_type"] = data["event_type"] - 1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)

    return data, mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth

def reset_indexes(dataframes):
    dataframes[0] = dataframes[0].reset_index(drop=True)
    dataframes[1] = dataframes[1].reset_index(drop=True)
    return dataframes

def unnormalize(x, mean, std):
    return x * std + mean

def one_hot_encoding_type(data):
    encoded_data = torch.zeros(data.shape[0], data.shape[1] + 2, dtype=torch.float32)
    encoded_data[:, 0] = data[:, 0]
    one_hot_order_type = torch.nn.functional.one_hot((data[:, 1]).to(torch.int64), num_classes=3).to(torch.float32)
    encoded_data[:, 1:4] = one_hot_order_type
    encoded_data[:, 4:] = data[:, 2:]
    return encoded_data

def tanh_encoding_type(data):
    data[:, 1] = torch.where(data[:, 1] == 1.0, 2.0, torch.where(data[:, 1] == 2.0, 1.0, data[:, 1]))
    data[:, 1] = data[:, 1] - 1
    return data

def to_sparse_representation(lob, n_levels):
    if not isinstance(lob, np.ndarray):
        lob = np.array(lob)
    sparse_lob = np.zeros(n_levels * 2)
    for j in range(lob.shape[0] // 2):
        if j % 2 == 0:
            ask_price = lob[0]
            current_ask_price = lob[j*2]
            depth = (current_ask_price - ask_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)] = lob[j*2+1]
        else:
            bid_price = lob[2]
            current_bid_price = lob[j*2]
            depth = (bid_price - current_bid_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)+1] = lob[j*2+1]
    return sparse_lob

def labeling(X, len, h, num_classes=3):
    assert len > 0, "Length must be greater than 0"
    assert h > 0, "Horizon must be greater than 0"
    
    if h < len:
        len = h
    
    print(f"Starting labeling with X shape: {X.shape}, len: {len}, h: {h}, num_classes: {num_classes}")
    # Process in batches to reduce memory usage
    batch_size = 10000  # Adjustable based on memory constraints
    n_samples = X.shape[0]
    labels = np.zeros(n_samples - h, dtype=np.int64)
    
    for i in range(0, n_samples - h, batch_size):
        print(f"Processing batch {i} to {min(i + batch_size, n_samples - h)}")
        start_idx = i
        end_idx = min(i + batch_size, n_samples - h)
        
        previous_ask_prices = np.lib.stride_tricks.sliding_window_view(X[start_idx:end_idx, 0], window_shape=len)[:-h]
        previous_bid_prices = np.lib.stride_tricks.sliding_window_view(X[start_idx:end_idx, 2], window_shape=len)[:-h]
        future_ask_prices = np.lib.stride_tricks.sliding_window_view(X[start_idx:end_idx, 0], window_shape=len)[h:]
        future_bid_prices = np.lib.stride_tricks.sliding_window_view(X[start_idx:end_idx, 2], window_shape=len)[h:]

        previous_mid_prices = (previous_ask_prices + previous_bid_prices) / 2
        future_mid_prices = (future_ask_prices + future_bid_prices) / 2

        previous_mid_prices = np.mean(previous_mid_prices, axis=1)
        future_mid_prices = np.mean(future_mid_prices, axis=1)

        percentage_change = (future_mid_prices - previous_mid_prices) / previous_mid_prices
        alpha = np.abs(percentage_change).mean() / 2
        
        print(f"Batch {i} alpha: {alpha}")
        if num_classes == 3:
            labels[start_idx:end_idx] = np.where(percentage_change < -alpha, 2, np.where(percentage_change > alpha, 0, 1))
        elif num_classes == 2:
            labels[start_idx:end_idx] = np.where(percentage_change > alpha, 0, 1)  # up: 0, stat/down: 1
        else:
            raise ValueError(f"Unsupported num_classes: {num_classes}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Number of labels: {dict(zip(unique_labels, counts))}")
    print(f"Percentage of labels: {counts / labels.shape[0]}")
    if num_classes == 3 and len(unique_labels) < 3:
        print(f"Warning: Only {len(unique_labels)} classes found in labels: {unique_labels}. Expected 3 classes (up, stat, down).")
    elif num_classes == 2 and len(unique_labels) < 2:
        print(f"Warning: Only {len(unique_labels)} classes found in labels: {unique_labels}. Expected 2 classes (up, stat/down).")
    
    print(f"Labeling complete for X shape: {X.shape}")
    return labels