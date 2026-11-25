import os
import sys
import io
import math

# Force UTF-8 encoding for stdout/stderr to prevent Windows cp1252 charmap errors
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
from torchvision import models

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "onnx_models")
output_dir = os.path.abspath(output_dir)
os.makedirs(output_dir, exist_ok=True)

print(f"[1/6] Exporting ResNet-50 to {output_dir}...")
try:
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        resnet,
        dummy_input,
        os.path.join(output_dir, "resnet50.onnx"),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )
    print("  -> ResNet-50 exported successfully!")
except Exception as e:
    print(f"  -> ResNet-50 export failed: {e}")

print(f"[2/6] Exporting MobileNetV3 Large to {output_dir}...")
try:
    mobnet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    mobnet.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        mobnet,
        dummy_input,
        os.path.join(output_dir, "mobnet.onnx"),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )
    print("  -> MobileNet-v3 exported successfully!")
except Exception as e:
    print(f"  -> MobileNet-v3 export failed: {e}")

print(f"[3/6] Exporting LSTM to {output_dir}...")
try:
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights):
            super().__init__()
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(hidden_dim, vocab_size)
            if tie_weights:
                self.embedding.weight = self.fc.weight
            self.init_weights()

        def forward(self, src, hidden):
            embedding = self.dropout(self.embedding(src))
            output, hidden = self.lstm(embedding, hidden)
            output = self.dropout(output)
            prediction = self.fc(output)
            return prediction, hidden

        def init_weights(self):
            init_range_emb = 0.1
            init_range_other = 1 / math.sqrt(self.hidden_dim)
            self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
            self.fc.weight.data.uniform_(-init_range_other, init_range_other)
            self.fc.bias.data.zero_()
            for i in range(self.num_layers):
                self.lstm.all_weights[i][0] = torch.FloatTensor(
                    self.embedding_dim, self.hidden_dim
                ).uniform_(-init_range_other, init_range_other)
                self.lstm.all_weights[i][1] = torch.FloatTensor(
                    self.hidden_dim, self.hidden_dim
                ).uniform_(-init_range_other, init_range_other)

    vocab_size = 29423
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 2
    dropout_rate = 0.2
    tie_weights = True
    batch_size = 1
    seq_length = 32

    lstm_model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights)
    lstm_model.eval()
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
    hidden_ = (torch.zeros(num_layers, batch_size, hidden_dim),
               torch.zeros(num_layers, batch_size, hidden_dim))

    torch.onnx.export(
        lstm_model,
        args=(inputs, hidden_),
        f=os.path.join(output_dir, "lstm.onnx"),
        opset_version=14,
        input_names=["input", "hidden"],
        output_names=["output", "hidden_out"],
        dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "hidden": {1: "batch_size"}},
        dynamo=False
    )
    print("  -> LSTM exported successfully!")
except Exception as e:
    print(f"  -> LSTM export failed: {e}")

print(f"[4/6] Exporting BERT to {output_dir}...")
try:
    from transformers import BertModel
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert.eval()
    
    class BertWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.embeddings = m.embeddings
            self.encoder = m.encoder
        def forward(self, input_ids):
            emb = self.embeddings(input_ids)
            enc = self.encoder(emb)
            return enc[0] if isinstance(enc, tuple) else enc.last_hidden_state

    wrapped_bert = BertWrapper(bert)
    dummy_ids = torch.randint(0, 30522, (1, 128), dtype=torch.int64)
    torch.onnx.export(
        wrapped_bert,
        dummy_ids,
        os.path.join(output_dir, "bert-base.onnx"),
        opset_version=14,
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch_size"}},
        dynamo=False
    )
    print("  -> BERT exported successfully!")
except Exception as e:
    print(f"  -> BERT export failed: {e}")

print(f"[5/6] Exporting YOLOv8n to {output_dir}...")
try:
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n.pt')
    exported_path = yolo_model.export(format='onnx', imgsz=640, dynamic=False)
    target_path = os.path.join(output_dir, "yolov8n.onnx")
    if os.path.exists(exported_path) and os.path.abspath(exported_path) != target_path:
        import shutil
        shutil.move(exported_path, target_path)
    print("  -> YOLOv8n exported successfully!")
except Exception as e:
    print(f"  -> YOLOv8n export failed: {e}")

print(f"[6/6] Exporting Simple Linear (FCN) to {output_dir}...")
try:
    class SimpleLinearModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)

    fcn = SimpleLinearModel(64, 128)
    fcn.eval()
    dummy_input = torch.randn(1, 64)
    torch.onnx.export(
        fcn,
        dummy_input,
        os.path.join(output_dir, "simple_linear_64_128.onnx"),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )
    print("  -> Simple Linear model exported successfully!")
except Exception as e:
    print(f"  -> Simple Linear model export failed: {e}")

print("\nAll models exported successfully!")
