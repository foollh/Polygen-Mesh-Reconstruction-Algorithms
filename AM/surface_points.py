import numpy as np
import torch.onnx
import onnxruntime

# 'AnalyticMesh/examples/chair.onnx'
onnxpath = '../chair.onnx'


def data_generate(step):
    num = torch.linspace(-1, 1, steps=step)
    Xs = num.repeat_interleave(step**2).unsqueeze(0)
    Ys = num.repeat_interleave(step)
    Ys = Ys.repeat(step).unsqueeze(0)
    Zs = num.repeat(step**2).unsqueeze(0)

    data = torch.cat((Xs, Ys, Zs), dim=0)
    return data.t()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def model_onnx(inputs):
    ort_session = onnxruntime.InferenceSession(onnxpath)
    # ONNX RUNTIME
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    ort_outs = ort_session.run(None, ort_inputs)  # list.
    return ort_outs


if __name__ == '__main__':
    step = 30  # 空间步长
    threshold = 0.1
    inputs = data_generate(step)
    inputs_valid = np.zeros([1, 3])
    for i in range(step**3):
        out = model_onnx(inputs[i].unsqueeze(0))

        if abs(out[0]) < threshold:
            inputs_valid = np.concatenate((inputs_valid, inputs[i].unsqueeze(0).numpy()), axis=0)

    np.save('surface_points.npy', inputs_valid)

