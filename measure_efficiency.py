import torch
from ptflops import get_model_complexity_info

# app.py에서 사용하는 모델 로딩 함수를 그대로 가져옵니다.
from network.models import model_selection

def measure_model_efficiency():
    """
    XceptionNet 모델의 파라미터 수와 연산량(FLOPs)을 측정합니다.
    """
    print("효율성 측정을 시작합니다...")

    # 1. 모델 초기화 (app.py와 동일한 방식)
    try:
        model = model_selection(modelname='xception', num_out_classes=2)
        model.eval()  # 모델을 평가 모드로 설정
        print("✅ 모델 클래스를 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"❌ 모델 클래스 로딩에 실패했습니다: {e}")
        return

    # 2. ptflops를 위한 표준 입력 해상도 정의
    input_resolution = (3, 299, 299)

    # 3. 모델 복잡도 계산
    try:
        macs, params = get_model_complexity_info(
            model,
            input_resolution,
            as_strings=True,
            print_per_layer_stat=False, # 각 레이어별 정보는 출력하지 않음
            verbose=False # 진행 과정 출력 비활성화
        )
        print("✅ 모델 복잡도 계산을 완료했습니다.")
    except Exception as e:
        print(f"❌ 모델 복잡도 계산 중 오류가 발생했습니다: {e}")
        return

    # 4. 결과 출력
    # ⭐️ [수정된 부분] ⭐️
    # '8.45 GMac' 같은 문자열에서 공백을 기준으로 나눠 숫자 부분('8.45')만 안전하게 추출합니다.
    try:
        gmac_value = float(macs.split()[0])
        gflops = gmac_value * 2
    except (ValueError, IndexError):
        print(f"❌ MACs 값 '{macs}'에서 숫자 변환에 실패했습니다.")
        return

    print("\n" + "="*40)
    print("     모델 효율성 분석 결과 (XceptionNet)")
    print("="*40)
    print(f"  - Input Resolution : {input_resolution}")
    print(f"  - Parameters       : {params}")
    print(f"  - MACs             : {macs}")
    print(f"  - Estimated GFLOPs : {gflops:.2f} GFLOPs")
    print("="*40)


if __name__ == '__main__':
    measure_model_efficiency()