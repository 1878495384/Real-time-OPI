# 视频振幅-相位联合重建系统 (展示衍射视频+相位视频) - 保存对比视频版本
import os
import math
import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model import StarNet  # 你自己的网络

# ====================== 设备与模型 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.backends.cudnn.benchmark = True  # 让 CuDNN 自动选最快的卷积实现

model = StarNet(base_dim=16, depths=[2, 2, 2, 2, 2]).to(device)
model.load_state_dict(torch.load('best_model_starnet(d2)22222.pth', map_location=device))
model.eval()


# ====================== Fresnel 角谱传递函数 (GPU 版) ======================
def build_fresnel_H_torch(shape, z, wavelength, device):
    """
    使用 PyTorch 在 GPU 上构建 Fresnel 角谱传递函数:
    H(fx, fy) = exp(-i*k*z) * exp(-i*pi*lambda*z*(fx^2 + fy^2))

    shape: (H, W)
    z: 传播距离 (m)
    wavelength: 波长 (m)
    """
    height, width = shape
    k = 2 * math.pi / wavelength

    # 频率坐标 (与 numpy 版逻辑一致)
    fx = torch.fft.fftfreq(width, d=1.0, device=device) * width  # [W]
    fy = torch.fft.fftfreq(height, d=1.0, device=device) * height  # [H]

    # FY[i,j] = fy[i], FX[i,j] = fx[j]
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')  # 形状 [H, W]

    FX = FX.to(torch.float32)
    FY = FY.to(torch.float32)

    # 先把两部分的实数相位合成一个 Tensor:
    # phase = -k*z - pi*lambda*z*(fx^2 + fy^2)
    phase = -k * z - math.pi * wavelength * z * (FX ** 2 + FY ** 2)  # float32 Tensor [H,W]

    # 再做一次复指数: H = exp(i * phase)
    H = torch.exp(1j * phase)  # complex64 Tensor [H,W] on device

    return H


# ====================== 角谱迭代算法 (全部在 GPU 上) ======================
@torch.no_grad()
def recover_random_phase_angular_spectrum_torch(image_t, diffraction_t,
                                                H, H_conj, n_iter=50):
    """
    利用角谱迭代算法 + GPU，从"原强度图 + 相位畸变图"恢复频域随机相位。

    输入:
        image_t        : torch.Tensor [H,W], float32, 在 device 上，原强度图 (物面)
        diffraction_t  : torch.Tensor [H,W], float32, 在 device 上，相位畸变图 (像面)
        H, H_conj      : Fresnel 传递函数及其共轭，complex64，[H,W]，在 device 上
        n_iter         : 迭代次数

    返回:
        random_phase_est_t : torch.Tensor [H,W]，float32，在 device 上，范围 [-pi, pi)
    """
    device = image_t.device

    # 归一化到 [0,1]
    image = image_t.clone()
    diffraction_image = diffraction_t.clone()

    im_max = image.max()
    if im_max > 0:
        image = image / im_max
    dif_max = diffraction_image.max()
    if dif_max > 0:
        diffraction_image = diffraction_image / dif_max

    # 物面场 U0 & 频域 F0
    U0 = image                           # 振幅 = image，相位 = 0
    F0 = torch.fft.fft2(U0)              # complex64
    B = torch.abs(F0) + 1e-12            # 频域幅度约束

    # 像面幅度约束 A_z
    A_z = diffraction_image

    # 初始化频域场 S: 幅度 = B，相位随机
    init_phase = 2 * math.pi * torch.rand_like(image)
    S = B * torch.exp(1j * init_phase)   # complex64

    for it in range(n_iter):
        # ---------- 频域 -> 像面 ----------
        Uz = torch.fft.ifft2(S * H)

        # 像面幅度约束：替换幅度为 A_z，保留相位
        Uz_phase = torch.angle(Uz)
        Uz_constrained = A_z * torch.exp(1j * Uz_phase)

        # ---------- 像面 -> 频域 ----------
        S_back = torch.fft.fft2(Uz_constrained) * H_conj

        # 频域幅度约束：|S| = B，保留相位
        S = B * torch.exp(1j * torch.angle(S_back))

    # 迭代结束，S_est ≈ F0 * exp(j * random_phase_est)
    S_est = S
    random_phase_est = torch.angle(S_est * torch.conj(F0))  # [-pi, pi)

    return random_phase_est  # 仍在 GPU 上


# ====================== 视频处理主函数 (主要计算在 GPU) ======================
def process_video_with_phase_recovery(model, input_video_path, groundtruth_video_path,
                                     z=0.1, wavelength=632.8e-9, n_iter=50, output_dir="videos"):
    """处理输入视频, 恢复每帧并重建随机相位, 实时显示三列 (主要计算在 GPU)"""
    os.makedirs(output_dir, exist_ok=True)

    # 打开输入视频和真值视频 (I/O 仍在 CPU)
    cap_input = cv2.VideoCapture(input_video_path)
    cap_gt = cv2.VideoCapture(groundtruth_video_path)

    if not cap_input.isOpened():
        print("无法打开输入视频文件")
        return
    if not cap_gt.isOpened():
        print("无法打开真值视频文件")
        return

    # 获取视频属性
    width = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_input.get(cv2.CAP_PROP_FPS)
    print(f"输入视频分辨率: {width}x{height}, 帧率: {fps}")

    # 目标尺寸 (网络与相位恢复使用)
    H_img, W_img = 512, 512

    # 在 GPU 上预先构建 Fresnel 传递函数 H / H_conj
    H_torch = build_fresnel_H_torch((H_img, W_img), z, wavelength, device=device)
    H_conj_torch = torch.conj(H_torch)

    # 设置输出视频
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_restored = os.path.join(output_dir, f"restored_{timestamp}.avi")
    output_phase = os.path.join(output_dir, f"phase_{timestamp}.avi")
    output_comparison = os.path.join(output_dir, f"comparison_{timestamp}.avi")  # 新增：对比视频

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_restored = cv2.VideoWriter(output_restored, fourcc, fps, (W_img, H_img), isColor=False)
    out_phase = cv2.VideoWriter(output_phase, fourcc, fps, (W_img, H_img), isColor=False)
    
    # 新增：创建一行三列对比视频的写入器
    separator_width = 5
    title_height = 50
    comparison_width = 3 * W_img + 2 * separator_width
    comparison_height = H_img + title_height
    out_comparison = cv2.VideoWriter(output_comparison, fourcc, fps, 
                                     (comparison_width, comparison_height), isColor=False)

    # 创建显示窗口
    cv2.namedWindow("Video Comparison", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Comparison", comparison_width, comparison_height)

    frame_count = 0
    total_processing_time = 0.0

    # 灰度化权重 (BGR -> Gray)
    gray_weights = torch.tensor([0.114, 0.587, 0.299], device=device).view(3, 1, 1)

    while cap_input.isOpened() and cap_gt.isOpened():
        ret_input, frame_input = cap_input.read()
        ret_gt, frame_gt = cap_gt.read()

        if not ret_input or not ret_gt:
            print("视频读取结束")
            break

        start_time = cv2.getTickCount()

        # ---------- 把 BGR 帧搬到 GPU，做灰度和 resize ----------
        # 输入视频帧
        frame_input_t = torch.from_numpy(frame_input).to(device=device, dtype=torch.uint8)  # [H,W,3]
        frame_input_t = frame_input_t.permute(2, 0, 1).float() / 255.0  # [3,H,W], B,G,R in OpenCV

        # 真值视频帧
        frame_gt_t = torch.from_numpy(frame_gt).to(device=device, dtype=torch.uint8)  # [H,W,3]
        frame_gt_t = frame_gt_t.permute(2, 0, 1).float() / 255.0  # [3,H,W]

        # 灰度化: Y = 0.114B + 0.587G + 0.299R
        gray_input_t = (frame_input_t * gray_weights).sum(dim=0, keepdim=True)  # [1,H,W]
        gray_gt_t = (frame_gt_t * gray_weights).sum(dim=0, keepdim=True)        # [1,H,W]

        # resize 到 512x512
        gray_input_t = F.interpolate(gray_input_t.unsqueeze(0), size=(H_img, W_img),
                                     mode='bilinear', align_corners=False).squeeze(0)  # [1,H,W]
        gray_gt_t = F.interpolate(gray_gt_t.unsqueeze(0), size=(H_img, W_img),
                                  mode='bilinear', align_corners=False).squeeze(0)     # [1,H,W]

        # 供显示的退化输入图 (CPU uint8)
        input_display = (gray_input_t.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        # ---------- 神经网络恢复 (GPU) ----------
        frame_tensor = gray_input_t.unsqueeze(0)  # [1,1,H,W]
        with torch.no_grad():
            restored_frame_t = model(frame_tensor)  # [1,1,H,W] (假设输出同尺寸)

        restored_frame_t = restored_frame_t.squeeze().clamp(0, 1)  # [H,W]
        restored_frame_clipped = (restored_frame_t * 255).to(torch.uint8).cpu().numpy()

        # ---------- 角谱迭代恢复随机相位 (GPU) ----------
        # 注意: 这里仍然沿用你原逻辑: gray_gt_t 为原图，gray_input_t 为畸变图
        random_phase_t = recover_random_phase_angular_spectrum_torch(
            gray_gt_t.squeeze(0),      # [H,W]
            gray_input_t.squeeze(0),   # [H,W]
            H_torch,
            H_conj_torch,
            n_iter=n_iter
        )  # [H,W], 在 GPU 上，范围 [-pi, pi)

        # 相位归一化到 [0,255]，转 CPU 写视频
        phase_norm_t = (random_phase_t + math.pi) / (2 * math.pi)  # [0,1]
        phase_uint8 = (phase_norm_t.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        # ---------- 计时 ----------
        end_time = cv2.getTickCount()
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        total_processing_time += processing_time

        # 保存单独的视频
        out_restored.write(restored_frame_clipped)
        out_phase.write(phase_uint8)

        # 创建分隔线
        separator = np.ones((H_img, separator_width), dtype=np.uint8) * 255

        # 一行三列: 退化、恢复、相位
        video_row = np.hstack((input_display, separator,
                               restored_frame_clipped, separator,
                               phase_uint8))

        # 创建标题行
        total_width = 3 * W_img + 2 * separator_width
        title_row = np.ones((title_height, total_width), dtype=np.uint8) * 225

        titles = ["Degraded Video", "Restored Video", "Recovered Phase"]
        positions = [W_img // 2, W_img + separator_width + W_img // 2,
                     2 * (W_img + separator_width) + W_img // 2]

        for title, pos in zip(titles, positions):
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 2)[0]
            text_x = pos - text_size[0] // 2
            cv2.putText(title_row, title, (text_x, 35),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, 0, 2)

        # 合并标题和视频行
        combined_frame = np.vstack((title_row, video_row))

        # 保存对比视频
        out_comparison.write(combined_frame)

        # 显示 (CPU)
        cv2.imshow("Video Comparison", combined_frame)

        frame_count += 1
        print(f"处理帧 {frame_count}, 当前FPS: {1 / processing_time:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap_input.release()
    cap_gt.release()
    out_restored.release()
    out_phase.release()
    out_comparison.release()  # 释放对比视频
    cv2.destroyAllWindows()

    # 统计信息
    if frame_count > 0:
        average_fps = frame_count / total_processing_time
        print(f"\n恢复视频已保存到: {output_restored}")
        print(f"相位视频已保存到: {output_phase}")
        print(f"对比视频已保存到: {output_comparison}")  # 新增提示
        print(f"总帧数: {frame_count}, 总处理时间: {total_processing_time:.2f} 秒")
        print(f"平均FPS: {average_fps:.2f}")
    else:
        print("未处理任何帧")


# ====================== 主函数 ======================
def main():
    input_video_path = 'test/teddy_dif-1.avi'      # 待修复视频路径 (退化视频)
    groundtruth_video_path = 'test/teddy_ori-1.avi'  # 真值视频路径

    # 物理参数
    z = 0.1              # 衍射距离 (m)
    wavelength = 632.8e-9  # 波长 (m)
    n_iter = 20          # 相位恢复迭代次数 (可调小提速)

    process_video_with_phase_recovery(
        model,
        input_video_path,
        groundtruth_video_path,
        z=z,
        wavelength=wavelength,
        n_iter=n_iter
    )


if __name__ == "__main__":
    main()




    



# # 一行二列
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2  # 导入 OpenCV 用于视频处理
# import os
# import datetime
# from model import StarNet  # 导入自定义的StarNet模型

# # 设置设备为GPU（如果可用）或CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = StarNet(base_dim=16, depths=[3, 3, 12, 5, 5]).to(device)  # 实例化支持512x512的StarNet模型
# model.load_state_dict(torch.load('best_model_starnet(d2).pth'))  # 加载预训练模型权重
# model.eval()  # 设置模型为评估模式

# def process_video(model, input_video_path, output_dir="videos"):
#     """处理输入视频，恢复每帧并保存为新视频，同时实时显示，并计算平均FPS"""
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 打开输入视频
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return

#     # 获取视频属性
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"输入视频分辨率: {width}x{height}, 帧率: {fps}")

#     # 检查分辨率是否为 2048x2048
#     if width != 2048 or height != 2048:
#         print("警告：视频分辨率不是 2048x2048，模型输入将调整为 512x512")

#     # 设置输出视频
#     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#     output_video_path = os.path.join(output_dir, f"restored_{timestamp}.avi")
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (512, 512), isColor=False)

#     # 创建显示窗口
#     separator_width = 5  # 分隔线宽度
#     title_height = 50  # 标题行高度
#     cv2.namedWindow("Video Comparison", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Video Comparison", 1024 + separator_width, 512 + title_height)  # 调整窗口大小以容纳标题和分隔线

#     frame_count = 0
#     total_processing_time = 0.0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("视频读取结束")
#             break

#         # 记录开始时间
#         start_time = cv2.getTickCount()

#         # 转换为灰度并调整大小为 512x512
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized_frame = cv2.resize(gray_frame, (512, 512)) / 255.0
#         original_display = cv2.resize(gray_frame, (512, 512))  # 用于显示的原帧

#         # 将帧转换为张量并添加批次维度
#         frame_tensor = torch.tensor(resized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

#         # 通过模型进行预测
#         with torch.no_grad():
#             restored_frame = model(frame_tensor)

#         # 将恢复的帧转换回 NumPy 数组
#         restored_frame = restored_frame.squeeze().cpu().numpy()
#         # 归一化到 [0, 255] 并转换为 uint8
#         restored_frame = (restored_frame * 255).clip(0, 255).astype(np.uint8)

#         # 记录结束时间并计算处理时间
#         end_time = cv2.getTickCount()
#         processing_time = (end_time - start_time) / cv2.getTickFrequency()
#         total_processing_time += processing_time

#         # 保存恢复帧到视频
#         out.write(restored_frame)

#         # 创建分隔线（白色垂直线）
#         separator = np.ones((512, separator_width), dtype=np.uint8) * 255

#         # 将原帧、分隔线和恢复帧水平合并为一行两列
#         video_row = np.hstack((original_display, separator, restored_frame))

        
#         # 创建灰色背景的标题行（128表示中灰色）
#         title_row = np.ones((title_height, 1024 + separator_width), dtype=np.uint8) * 225
        
#         # 定义标题文本
#         original_text = "Original Video"
#         restored_text = "Restored Video"
        
#         # 计算每个标题的位置使其居中显示
#         # 获取文本尺寸
#         original_text_size = cv2.getTextSize(original_text, cv2.FONT_HERSHEY_TRIPLEX, 1, 2)[0]
#         restored_text_size = cv2.getTextSize(restored_text, cv2.FONT_HERSHEY_TRIPLEX, 1, 2)[0]
        
#         # 计算左半部分和右半部分的中心位置
#         left_center_x = (512) // 2  # 原视频区域的中心x坐标
#         right_center_x = 512 + separator_width + (512) // 2  # 重建视频区域的中心x坐标
        
#         # 计算文本起始位置（使文本在各自区域居中）
#         original_x = left_center_x - original_text_size[0] // 2
#         restored_x = right_center_x - restored_text_size[0] // 2
        
#         # 在标题行上添加标题（使用罗马字体FONT_HERSHEY_TRIPLEX，黑色文字）
#         cv2.putText(title_row, original_text, (original_x, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, 0, 2)
#         cv2.putText(title_row, restored_text, (restored_x, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, 0, 2)

        
#         # 将标题行和视频行垂直合并
#         combined_frame = np.vstack((title_row, video_row))

#         # 显示合并后的帧
#         cv2.imshow("Video Comparison", combined_frame)

#         frame_count += 1
#         print(f"处理帧 {frame_count}, 当前FPS: {1 / processing_time:.2f}")

#         # 按 'q' 键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 释放资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # 计算并打印平均FPS
#     if frame_count > 0:
#         average_fps = frame_count / total_processing_time
#         print(f"恢复视频已保存到: {output_video_path}")
#         print(f"总帧数: {frame_count}, 总处理时间: {total_processing_time:.2f} 秒")
#         print(f"平均FPS: {average_fps:.2f}")
#     else:
#         print("未处理任何帧")

# def main():
#     """主函数：处理输入视频并恢复"""
#     input_video_path = 'test/9.avi'  # 替换为你的输入视频路径
#     process_video(model, input_video_path)

# if __name__ == "__main__":
#     main()











    

# # 计算fps
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2  # 导入 OpenCV 用于视频处理
# import os
# import datetime
# from model import StarNet  # 导入自定义的StarNet模型

# # 设置设备为GPU（如果可用）或CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = StarNet(base_dim=16, depths=[3, 3, 12, 5, 5]).to(device)  # 实例化支持512x512的StarNet模型
# model.load_state_dict(torch.load('best_model_starnet(d2).pth'))  # 加载预训练模型权重
# model.eval()  # 设置模型为评估模式

# def process_video(model, input_video_path, output_dir="videos"):
#     """处理输入视频，恢复每帧并保存为新视频，同时实时显示，并计算平均FPS"""
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 打开输入视频
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return

#     # 获取视频属性
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"输入视频分辨率: {width}x{height}, 帧率: {fps}")

#     # 检查分辨率是否为 2048x2048
#     if width != 2048 or height != 2048:
#         print("警告：视频分辨率不是 2048x2048，模型输入将调整为 512x512")

#     # 设置输出视频
#     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#     output_video_path = os.path.join(output_dir, f"restored_{timestamp}.avi")
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (512, 512), isColor=False)

#     # 创建显示窗口
#     cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Original Video", 512, 512)
#     cv2.namedWindow("Restored Video", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Restored Video", 512, 512)

#     frame_count = 0
#     total_processing_time = 0.0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("视频读取结束")
#             break

#         # 记录开始时间
#         start_time = cv2.getTickCount()

#         # 转换为灰度并调整大小为 512x512
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized_frame = cv2.resize(gray_frame, (512, 512)) / 255.0

#         # 将帧转换为张量并添加批次维度
#         frame_tensor = torch.tensor(resized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

#         # 通过模型进行预测
#         with torch.no_grad():
#             restored_frame = model(frame_tensor)

#         # 将恢复的帧转换回 NumPy 数组
#         restored_frame = restored_frame.squeeze().cpu().numpy()
#         # 归一化到 [0, 255] 并转换为 uint8
#         restored_frame = (restored_frame * 255).clip(0, 255).astype(np.uint8)

#         # 记录结束时间并计算处理时间
#         end_time = cv2.getTickCount()
#         processing_time = (end_time - start_time) / cv2.getTickFrequency()
#         total_processing_time += processing_time

#         # 保存恢复帧到视频
#         out.write(restored_frame)

#         # 显示原帧和恢复帧
#         cv2.imshow("Original Video", cv2.resize(gray_frame, (512, 512)))
#         cv2.imshow("Restored Video", restored_frame)

#         frame_count += 1
#         print(f"处理帧 {frame_count}, 当前FPS: {1 / processing_time:.2f}")

#         # 按 'q' 键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 释放资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # 计算并打印平均FPS
#     if frame_count > 0:
#         average_fps = frame_count / total_processing_time
#         print(f"恢复视频已保存到: {output_video_path}")
#         print(f"总帧数: {frame_count}, 总处理时间: {total_processing_time:.2f} 秒")
#         print(f"平均FPS: {average_fps:.2f}")
#     else:
#         print("未处理任何帧")

# def main():
#     """主函数：处理输入视频并恢复"""
#     input_video_path = 'test/9.avi'  # 替换为你的输入视频路径
#     process_video(model, input_video_path)

# if __name__ == "__main__":
#     main()






# # 计算fps  331255
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2  # 导入 OpenCV 用于视频处理
# import os
# import datetime
# from model import StarNet  # 导入自定义的StarNet模型

# # 设置设备为GPU（如果可用）或CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = StarNet(base_dim=16, depths=[3, 3, 12, 5, 5]).to(device)  # 实例化支持512x512的StarNet模型
# model.load_state_dict(torch.load('best_model_starnet(d3).pth'))  # 加载预训练模型权重
# model.eval()  # 设置模型为评估模式

# def process_video(model, input_video_path, output_dir="videos"):
#     """处理输入视频，恢复每帧并保存为新视频，同时实时显示，并计算平均FPS"""
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 打开输入视频
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return

#     # 获取视频属性
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"输入视频分辨率: {width}x{height}, 帧率: {fps}")

#     # 检查分辨率是否为 2048x2048
#     if width != 2048 or height != 2048:
#         print("警告：视频分辨率不是 2048x2048，模型输入将调整为 512x512")

#     # 设置输出视频
#     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#     output_video_path = os.path.join(output_dir, f"restored_{timestamp}.avi")
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (512, 512), isColor=False)

#     # 创建显示窗口
#     cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Original Video", 512, 512)
#     cv2.namedWindow("Restored Video", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Restored Video", 512, 512)

#     frame_count = 0
#     total_processing_time = 0.0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("视频读取结束")
#             break

#         # 记录开始时间
#         start_time = cv2.getTickCount()

#         # 转换为灰度并调整大小为 512x512
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized_frame = cv2.resize(gray_frame, (512, 512)) / 255.0

#         # 将帧转换为张量并添加批次维度
#         frame_tensor = torch.tensor(resized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

#         # 通过模型进行预测
#         with torch.no_grad():
#             restored_frame = model(frame_tensor)

#         # 将恢复的帧转换回 NumPy 数组
#         restored_frame = restored_frame.squeeze().cpu().numpy()
#         # 归一化到 [0, 255] 并转换为 uint8
#         restored_frame = (restored_frame * 255).clip(0, 255).astype(np.uint8)

#         # 记录结束时间并计算处理时间
#         end_time = cv2.getTickCount()
#         processing_time = (end_time - start_time) / cv2.getTickFrequency()
#         total_processing_time += processing_time

#         # 保存恢复帧到视频
#         out.write(restored_frame)

#         # 显示原帧和恢复帧
#         cv2.imshow("Original Video", cv2.resize(gray_frame, (512, 512)))
#         cv2.imshow("Restored Video", restored_frame)

#         frame_count += 1
#         print(f"处理帧 {frame_count}, 当前FPS: {1 / processing_time:.2f}")

#         # 按 'q' 键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 释放资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # 计算并打印平均FPS
#     if frame_count > 0:
#         average_fps = frame_count / total_processing_time
#         print(f"恢复视频已保存到: {output_video_path}")
#         print(f"总帧数: {frame_count}, 总处理时间: {total_processing_time:.2f} 秒")
#         print(f"平均FPS: {average_fps:.2f}")
#     else:
#         print("未处理任何帧")

# def main():
#     """主函数：处理输入视频并恢复"""
#     input_video_path = 'test/11-1.avi'  # 替换为你的输入视频路径
#     process_video(model, input_video_path)

# if __name__ == "__main__":
#     main()





# # 对视频进行恢复测试
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2  # 导入 OpenCV 用于视频处理
# import os
# import datetime
# from model import StarNet  # 导入自定义的StarNet模型

# # 设置设备为GPU（如果可用）或CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = StarNet(base_dim=16, depths=[3, 3, 12, 5, 5]).to(device)  # 实例化支持512x512的StarNet模型
# model.load_state_dict(torch.load('best_model_starnet(d2).pth'))  # 加载预训练模型权重
# model.eval()  # 设置模型为评估模式

# def process_video(model, input_video_path, output_dir="videos"):
#     """处理输入视频，恢复每帧并保存为新视频，同时实时显示"""
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 打开输入视频
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return

#     # 获取视频属性
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"输入视频分辨率: {width}x{height}, 帧率: {fps}")

#     # 检查分辨率是否为 2048x2048
#     if width != 2048 or height != 2048:
#         print("警告：视频分辨率不是 2048x2048，模型输入将调整为 512x512")

#     # 设置输出视频
#     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#     output_video_path = os.path.join(output_dir, f"restored_{timestamp}.avi")
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (512, 512), isColor=False)

#     # 创建显示窗口
#     cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Original Video", 512, 512)
#     cv2.namedWindow("Restored Video", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Restored Video", 512, 512)

#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("视频读取结束")
#             break

#         # 转换为灰度并调整大小为 512x512
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         resized_frame = cv2.resize(gray_frame, (512, 512)) / 255.0

#         # 将帧转换为张量并添加批次维度
#         frame_tensor = torch.tensor(resized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

#         # 通过模型进行预测
#         with torch.no_grad():
#             restored_frame = model(frame_tensor)

#         # 将恢复的帧转换回 NumPy 数组
#         restored_frame = restored_frame.squeeze().cpu().numpy()
#         # 归一化到 [0, 255] 并转换为 uint8
#         restored_frame = (restored_frame * 255).clip(0, 255).astype(np.uint8)

#         # 保存恢复帧到视频
#         out.write(restored_frame)

#         # 显示原帧和恢复帧
#         cv2.imshow("Original Video", cv2.resize(gray_frame, (512, 512)))
#         cv2.imshow("Restored Video", restored_frame)

#         frame_count += 1
#         print(f"处理帧 {frame_count}")

#         # 按 'q' 键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 释放资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"恢复视频已保存到: {output_video_path}")

# def main():
#     """主函数：处理输入视频并恢复"""
#     input_video_path = 'test/9.avi'  # 替换为你的输入视频路径
#     process_video(model, input_video_path)

# if __name__ == "__main__":
#     main()