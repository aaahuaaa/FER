import torch
import sys
sys.path.append(".")  # 添加当前目录到Python路径
from networks.dan import DAN


def convert_pth2onnx(pth_path, output_name, num_class=8, num_head=4):
    # 初始化模型
    model = DAN(num_class=num_class, num_head=num_head)

    # 加载预训练权重
    checkpoint = torch.load(pth_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建虚拟输入（需与训练时输入尺寸一致）
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        output_name,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    print(f"ONNX模型已保存至 {output_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, required=True, help='输入.pth模型路径')
    parser.add_argument('--out', type=str, default='model.onnx', help='输出ONNX路径')
    parser.add_argument('--num_class', type=int, default=7, help='类别数（需与训练时一致）')
    parser.add_argument('--num_head', type=int, default=4, help='注意力头数')
    args = parser.parse_args()

    convert_pth2onnx(args.pth, args.out, args.num_class, args.num_head)
