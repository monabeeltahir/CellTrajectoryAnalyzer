import cv2

def setup_gpu(use_gpu: bool):
    if not use_gpu:
        return False

    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        print(f"✅ OpenCL enabled: {cv2.ocl.useOpenCL()}")
        print(f"OpenCL Device: {cv2.ocl.Device.getDefault().name()}")
        return True

    print("⚠️ OpenCL not available, falling back to CPU")
    return False


def create_preprocessors(config):
    fgbg = cv2.createBackgroundSubtractorKNN(
        history=config.history,
        dist2Threshold=config.dist2_threshold,
        detectShadows=config.detect_shadows
    )
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size
    )
    return fgbg, clahe


def preprocess_frame(frame, fgbg, clahe, config):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    fgmask = fgbg.apply(enhanced)
    blurred = cv2.GaussianBlur(fgmask, config.blur_kernel, 0)
    _, thresh = cv2.threshold(blurred, config.threshold_value, 255, cv2.THRESH_BINARY)
    return gray, enhanced, fgmask, blurred, thresh