# 데이콘 펭귄 몸무게 예측 경진대회

![image](https://user-images.githubusercontent.com/40455392/147430610-a1e9883d-f19c-4143-b102-bf686e3607be.png)



-  현재 스코어 : 334.26565 (242명 중 21 등)
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



## 4. 후기

대회 종료 이후 작성 예정입니다.