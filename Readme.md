# 데이콘 펭귄 몸무게 예측 경진대회

![image](https://user-images.githubusercontent.com/40455392/148344581-79057de0-29a2-423e-986e-3661de4c913f.png)



-  현재 스코어 : 284.3466 (684명 중 55등)
-  대회 링크 :  [링크](https://dacon.io/competitions/official/235862/overview/description)



## 1.  프로젝트 개요

- 데이콘에서 제공하는 베이직 경진대회인 `펭귄 몸무게 예측` 하는 모델을 생성해 결과 값을 제출합니다.

- Metric은 RMSE로, 실제 값과 예측 값 차이만큼의 제곱 합의 제곱근을 지표로 사용합니다.

  ![image](https://user-images.githubusercontent.com/40455392/147523354-3017bf03-9ccc-49b5-bb09-7cce17d0033c.png)

- 기본 대회이고, 주어진 데이터도 소량이기 때문에 큰 의미를 갖는 대회는 아닙니다.

- 그러나 이 대회를 통해 2가지를 사용해서 성과를 만들고 싶었습니다.

  - 부스팅 모델의 앙상블 (XGBoost, LightGBM, CatBoost)
  - 데이터 스케일 조정 후 모델 학습 시, 학습 양상 추적

- 위에 언급한 2가지가 갖는 의미는 다음과 같은 2가지 의미를 내포합니다.

  - 여러 모델을 조합해 가장 높은 학습 데이터를 갖는 모델의 정보를 사용해 보다 뛰어난 성능을 만들어 낸다
    - 이른바 일반 대중의 지혜를 이용한다. (마치 위키피디아처럼)
  - 사이킷런에서 제공하는 데이터의 스케일 조정 방법이 여러개 있습니다.
  - 데이터의 스케일을 조정함으로써 학습 속도를 조정 전보다 압도적으로 향상시킬 수 있었습니다.

  ```python
  from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
  
  continuous_names = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)']
  
  # MinMaxScaler -> 2번째로 성능이 좋음
  # scaler = MinMaxScaler()
  
  # scaler = StandardScaler()
  
  # scaler = RobustScaler()
  
  # Normalizer -> 단순 정규화만으로는 높은 성능 향상을 기대하기 어려움
  # scaler = Normalizer()
  
  # scaler = QuantileTransformer()
  
  # QuantileTransformer에 정규화 적용 -> 성능상 큰 차이가 보이지 않음 (아마 회귀 데이터를 다루기 때문에 큰 차이가 없는 것으로 보임)
  # scaler = QuantileTransformer(output_distribution = 'normal')
  
  # PowerTransformer -> 가장 성능이 좋음
  scaler = PowerTransformer()
  
  def scale(df, columns):
      train_scaler = scaler.fit_transform(df[columns])
      df[columns] = pd.DataFrame(data=train_scaler, columns=columns)
      
      return df
  ```

  

- 결측 값과 카테고리형 칼럼의 원활한 모델 학습을 위해 판다스에서 제공하고 있는 `get_dummies`  메소드를 이용해 원-핫 인코딩을 수행했습니다.

  - 기존에 `LabelEncoder`를 이용하면 정수형 인코딩이 가능하지만, 특정 수들 간의 거리가 가까울 수록 모델이 데이터 사이의 연관성이 있다고 오판할 수 있습니다.
    - 예를 들어, 0 / 1 사이가 가까이 있다고 해서 칼럼 사이의 특별한 상관 관계가 있는 것은 아닌 경우
  - 따라서, 이러한 상황을 회피하기 위해 get_dummies를 사용했고, 소정의 RMSE 점수를 향상시킬 수 있었습니다.

  ```python
  # 결측치를 처리하는 함수를 작성합니다.
  def handle_na(data, missing_col):
      temp = data.copy()
      for col, dtype in missing_col:
          if dtype == 'O':
              # 카테고리형 feature가 결측치인 경우 해당 행 및 카테고리형 데이터 원-핫 인코딩 수행
              temp = pd.get_dummies(temp)
          elif dtype == int or dtype == float:
              # 수치형 feature가 결측치인 경우 평균값을 채워주었습니다.
              temp.loc[:,col] = temp[col].fillna(temp[col].mean())
      return temp
  
  data = handle_na(df, missing_col)
  
  data.head()
  ```



## 2. 모델 선택 이유

- 최근 TF 또는 Torch를 이용한 딥러닝 모델을 사용하는 것이 아닌 부스팅 모델을 사용하는 경우가 잦습니다.
- 딥러닝 코드를 짜서 좋은 성능을 내는 것도 좋은 방법이지만, 딥러닝은 만능이 아닙니다.
- 머신러닝이라는 범주 내에서 좋은 성능을 낼 수 있는 여러 모델들을 두루 경험하는 것이 수 많은 삽질을 요구하는 머신러닝 분야에서 요구하는 역량 중 하나일 것이고, 이 같은 관점에서 사람들을 통해 충분히 검증되어 이용되고 있는 부스팅 모델들을 필두로 경험을 늘려가고 있습니다.
  - 바로 얼마 전까지는 Logistic & Ridge & Lasso 회귀 함수를 이용해 성능 검증하는 기간을 가졌습니다.
- LightGBM은 XGBoost보다 경량화 되어 있기 때문에 실제 프로덕트를 배포한다면 어떨까? 라는 관점에서 이용하게 되었습니다.



## 3. Data-Centric 관점에서의 대회 접근

- 솔직히 소규모 데이터셋의 기본 성능 점검 대회이기 때문에 데이터 중심으로 학습하는 것에 대해 간과하고 있었습니다.
- 그러나 여전히 답은 데이터에 있었고, 범주형 데이터들의 라벨이 몇 가지 안되었고 Missing Value조차 존재했기 때문에 `get_dummies`를 이용한 원핫 인코딩 접근은 데이터 중심 사고에서 옳은 방향이었다고 생각합니다.
- 연속형 데이터의 경우, 데이터 별 편차가 다양했기 때문에 스케일 조정이 필요하다고 생각했습니다.
- 간단하게 이용해 본 것은 `StandardScaler`이나, 이것 밖에 방법이 없나 싶어 다양한 전처리 툴을 찾아보고 `PowerTransformer`를 찾아 이용해보게 되었습니다.
  - 늘 사용하던 것만 사용하는 것은 굉장히 위험합니다.
  - 트렌드와 새롭게 만들어진 것들에 대한 성능 검증을 해보는 습관을 충분히 가져봐야 더 좋은 모델을 만드는 것에 기여할 수 있습니다.



## 4. Pycaret 사용

- 다른 사람의 베이스 코드를 보다가 pycaret이라는 것을 알게 되었습니다.
- 한 번에 여러 머신 러닝 모델을 돌리면서 데이터에 최적화 되어 있는 모델을 이용할 수 있는 모듈인데, 여러 모듈을 조합하는 것도 매우 편리합니다.
- 자주 애용해야겠습니다. 여러 의미로 앙상블을 하는 것과 최적의 하이퍼 파라미터를 찾아내는 것 모두 사람의 감에 의존하기 보다 컴퓨터의 연산 값이 보다 정확하기 때문입니다.
- 최종스코어가 매우 많이 올랐고, 현재 26위가 되었습니다.

![image](https://user-images.githubusercontent.com/40455392/147849659-68d13c42-cddb-4021-9d1c-0fe79b60f8c3.png)



## 4. 최종 결과

![image](https://user-images.githubusercontent.com/40455392/148342298-0b0611ae-da52-4346-b588-1d853ee73cf5.png)

- 분명히 다양한 데이터 처리 도구와 모델 결합 등의 모듈을 사용하는 방식이 꾸준히 늘어나고 있습니다.
- 하지만 전반적으로 어떻게 해야 최고의 성능을 뽑아내는 모델과 전처리 방식이 있는지에 대해 가늠하기가 어렵습니다.
- 여기에는 여러 이유가 있겠지만, 이 프로젝트를 마감하는 현재 시점에서 보건데, 다음과 같은 3가지 사유가 있을 것으로 보입니다.
  - 각 프로젝트 또는 대회에서 제공되는 데이터에 대한 이해 부족
  - 이에 따라 데이터에 맞는 적합한 전처리 모듈 등을 사용하지 못하는 상황 발생
  - 성능을 높일 수 있는 여러 스킬들에 대한 이해 부족 (앙상블, 모델, 하이퍼 파라미터 등)
- 이번에는 머신러닝 기법 중 `pycaret`에 대해 알게 되었고, 프로젝트가 간단했기 때문에 비교적 적용이 쉬웠습니다.
- 그러나 딥러닝의 경우 검색을 해봤지만, `pycaret` 처럼 데이터에 대한 최적의 모델 정보를 알려주는 모듈은 아직 없는 것 같고, 모델에 대한 어느 정도 이해를 바탕으로 조합해야하는 상황에 직면할 것입니다. (검색 능력이 부족해 관련 모듈이 있음에도 찾지 못한 것일 수 있습니다.)
- 위에도 언급이 되었지만, 때문에 익숙한 모델, 전처리 도구만 사용하면 안됩니다.
- 늘 모델 트렌드와 데이터 가공 테크닉 성향에 대해 촉각을 곤두세우고 간단한 데이터에라도 적용할 필요가 있습니다.

![image](https://user-images.githubusercontent.com/40455392/148342452-10f0e3cf-4e3c-4b9b-9a53-156b1846b5cd.png)



## 5. 후기

- 데이콘의 경우, 한국에서 진행하고 있는 데이터 대회라서 꾸준히 상위 10% 안에는 들고 있습니다.
- 그러나 결과적으로 Kaggle 등의 세계 속에서 상위권을 유지하는 것이 더 중요하고, 다양한 문화권과 사람들의 테크닉을 받아들일 필요가 있습니다.
- `그 때는 맞고, 지금은 틀리다.` 또는 `그 때는 틀리고, 지금은 맞다.` 라는 말이 머신러닝 프로젝트에 가장 잘 들어맞는 것 같습니다.
- Kaggle TPS 대회에선, `MinMaxScaler` 를 이용해서 데이터 전처리를 했을 때, 더 좋은 성능이 나왔는데, 이 대회에서는 그렇지 않았습니다
  - 전반적으로 Scaler를 적용했을 때 더 성능이 나빠지는 것을 볼 수 있었습니다.
  - Boost 모델을 결합하면, 회귀 모델을 결합했을 때보다 성능이 나빴습니다.
- 모델을 적용하기 전에 raw data와 preprocessing data 모두 시각화를 충분히 시켜보면서 데이터를 다각도로 볼 수 있는 것이 굉장히 중요합니다.
- 꾸준히 다른 사람들 코드도 볼 수 있기 때문에 대회를 참여하는 것은 실력 향상이라는 측면에서 많은 의의가 있습니다.