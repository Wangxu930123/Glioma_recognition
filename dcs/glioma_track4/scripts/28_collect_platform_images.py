"""把三份官方指南里的关键截图整理进工程，供《平台使用指南》引用。

挑选依据是提取出的「图文对应关系」——每张图都能定位到它解释的那一步，
而不是把图堆在一起。文件名按"环境_动作"重命名，便于对照。
"""
from __future__ import annotations

import pathlib
import shutil

_HERE = pathlib.Path(__file__).resolve().parent
_TRACK4 = _HERE.parent
SRC = _TRACK4 / "docs" / ".platform_raw"           # 由 27_extract_platform_docs.py 产出
DST = _TRACK4 / "docs" / "platform_images"
DST.mkdir(parents=True, exist_ok=True)

PC = SRC / "云电脑平台使用指南V1.0_imgs"
PT = SRC / "训推平台使用指南V1.0_imgs"
CM = SRC / "赛事管理平台使用指南V2.0_imgs"
FR = SRC / "frames"

#: (源文件, 目标名, 说明) —— 说明来自官方文档中该图所在段落的原文
ITEMS = [
    # ---- 赛事管理平台 ----
    (FR / "f001.jpg", "01_赛事平台_个人工作台.jpg",
     "赛事管理平台「个人工作台」：常用功能入口（文件资料/问题咨询/申诉/个人空间/我的资源）与赛段任务"),
    (CM / "image6.png", "02_赛事平台_文件资料.png",
     "路径：赛事管理平台 → 个人工作台 → 常用功能 → 文件资料（下载云桌面安装包与参赛指南）"),
    (CM / "image1.png", "03_赛事平台_验证测评.png",
     "路径：官网 → 赛事管理平台 → 个人工作台 → 初赛阶段 → 验证测评（每队每赛道每天 5 次）"),
    (CM / "image3.png", "04_赛事平台_初赛测评.png",
     "路径：初赛阶段 → 初赛测评（每赛道**只发起一次**，发起后冻结容器云服务部署权限）"),
    # ---- 云桌面客户端 ----
    (PC / "image1.png", "05_云桌面_安装向导.png",
     "双击 exe 安装包 → 下一步 → 选择安装目录 → 安装（安装前务必退出 360 等杀毒软件）"),
    (PC / "image6.png", "06_云桌面_桌面快捷方式.png",
     "安装完成后桌面出现「大赛云桌面」快捷方式"),
    (PC / "image8.png", "07_云桌面_填服务地址.png",
     "服务地址填 https://ybystds.ybj.gxzf.gov.cn:30037，点【下一步】"),
    (PC / "image9.png", "08_云桌面_登录.png",
     "账号密码与**赛事管理平台注册的一致**，点【登录】"),
    (PC / "image11.png", "09_云桌面_连接桌面.png",
     "在桌面卡片页找到自己的桌面，点【连接】（首次登录绑定需 5-10 秒，可点右上角刷新）"),
    (PC / "image21.png", "10_云桌面_开机.png",
     "桌面处于「已关机」时：卡片右上角「…」→ 开机（关机期间无法连接）"),
    (PC / "image24.png", "11_云桌面_悬浮球工具栏.png",
     "连接后的悬浮球工具栏：状态监测 / 全屏切换 / 断开"),
    # ---- 训推平台 ----
    (PT / "image1.png", "12_训推_基础镜像.png",
     "路径：AI资源 → 镜像仓库 → 基础镜像（pytorch:2.8.0-cuda12.8-cudnn9-runtime 等）"),
    (PT / "image2.png", "13_训推_公共数据集.png",
     "路径：AI资产 → 存储与数据服务 → 公共存储 → public_dataset_{赛道}（**只读**挂载）"),
    (PT / "image5.png", "14_训推_私有存储三目录.png",
     "队伍私有存储默认三目录：answer（推理结果）/ train_model（模型）/ common（自定义）"),
    (PT / "image7.png", "15_训推_调度配置.png",
     "路径：算力调度 → AI资产 → 调度配置（可自定义算力规格，供创建实例时选用）"),
    (PT / "image15.png", "16_训推_添加实例.png",
     "路径：AI开发 → 容器实例 → 添加实例（**每队每赛道最多 4 个**）"),
    (PT / "image17.png", "17_训推_挂载存储.png",
     "创建实例时挂载：公共存储（只读）+ 我的存储（读写）；容器内路径 /2026aicompetition/workspace"),
    (PT / "image19.png", "18_训推_Web控制台入口.png",
     "实例创建成功后点「Web 连接」进入控制台"),
    (PT / "image21.png", "19_训推_VSCode_SSH.png",
     "访问方式 → 更多方式：复制服务器 IP、端口与 SSH 初始密码，用 VSCode SSH 或 ssh 命令连接"),
    (PT / "image29.png", "20_训推_测评容器.png",
     "训练完成后创建「测评容器」（**单租户最多 1 个**）：必须开 8000 端口且 /health 可访问"),
    # ---- 视频关键帧 ----
    (FR / "f004.jpg", "21_代码管理_Codeup.png",
     "云桌面内的「代码管理 Codeup」：URL 导入代码库（视频演示）"),
    (FR / "f007.jpg", "22_容器内_装依赖.png",
     "从云桌面 SSH 进容器后，在 /2026aicompetition/workspace/dcs/Glioma_recognition 下装依赖（视频演示）"),
]

ok, miss = 0, []
for src, dst_name, _desc in ITEMS:
    if not src.is_file():
        miss.append(str(src))
        continue
    shutil.copy2(src, DST / dst_name)
    ok += 1

print(f"复制 {ok} 张关键截图 → {DST}")
if miss:
    print("缺失：")
    for m in miss:
        print(f"  {m}")

# 同时把"全部截图"也留一份，便于人工查阅
raw = DST / "_原始截图"
raw.mkdir(exist_ok=True)
for tag, d in (("云电脑", PC), ("训推平台", PT), ("赛事平台", CM), ("视频帧", FR)):
    tgt = raw / tag
    tgt.mkdir(exist_ok=True)
    for f in sorted(d.iterdir()):
        if f.is_file():
            shutil.copy2(f, tgt / f.name)
print(f"全部原始截图 → {raw}（{sum(1 for _ in raw.rglob('*') if _.is_file())} 个文件）")
