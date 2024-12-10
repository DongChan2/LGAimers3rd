import numpy as np 
def PSFA(pred, target): 
    PSFA = 1
    for cat in range(5):
        ids = indexs_bigcat[cat]
        for day in range(21):
            total_sell = np.sum(target[ids, day]) # day별 총 판매량
            pred_values = pred[ids, day] # day별 예측 판매량
            target_values = target[ids, day] # day별 실제 판매량
            
            # 실제 판매와 예측 판매가 같은 경우 오차가 없는 것으로 간주 
            denominator = np.maximum(target_values, pred_values)
            diffs = np.where(denominator!=0, np.abs(target_values - pred_values) / denominator, 0)
            
            if total_sell != 0:
                sell_weights = target_values / total_sell  # Item별 day 총 판매량 내 비중
            else:
                sell_weights = np.ones_like(target_values) / len(ids)  # 1 / len(ids)로 대체
                
            if not np.isnan(diffs).any():  # diffs에 NaN이 없는 경우에만 PSFA 값 업데이트
                PSFA -= np.sum(diffs * sell_weights) / (21 * 5)
            
            
    return PSFA
        