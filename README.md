# 유방암 데이터 분석 (의사 결정 트리)

본 문서는 Kaggle에서 제공한 유방암 데이터셋을 활용하여 데이터 분석 및 의사 결정 트리를 통한 분류 모델 구축을 목표로 합니다. 이 데이터셋은 유방암 진단 정보를 담고 있습니다. 데이터를 적절하게 전처리한 후, 모델링과 예측을 수행합니다.

## 목차
1. [패키지 설치 및 그래프 설정](#1-패키지-설치-및-그래프-설정)
2. [데이터 수집](#2-데이터-수집)
3. [데이터 전처리](#3-데이터-전처리)
    1. [이상값 처리](#3-1-이상값-처리)
    2. [데이터 변수의 형 변환 및 삭제](#3-2-데이터-변수의-형-변환-및-삭제)
4. [의사 결정 트리 분석](#4-의사-결정-트리-분석)
    1. [데이터 분할](#4-1-데이터-분할)
    2. [의사 결정 트리 모델 학습](#4-2-의사-결정-트리-모델-학습)
    3. [모델 시각화](#4-3-모델-시각화)
    4. [가지치기 (Pruning)](#4-4-가지치기pruning)

5. [모델 평가](#5-모델-평가)
    1. [예측](#5-1-예측)
    2. [혼돈 메트릭스 (Confusion Matrix)](#5-2-혼돈-메트릭스confusion-matrix)
6. [결론](#6-결론)





## 1. 패키지 설치 및 그래프 설정

프로젝트를 시작하기 전, 필요한 R 패키지를 설치하고 그래프 설정을 합니다.
 ```
#패키지 부착, 출력 그래프의 크기를 설정
install.packages(c("tidyverse","data.table","caret"))
library(tidyverse)
library(data.table)
library(repr)

options(repr.plot.width=7,repr.plot.height=7)
```

## 2. 데이터 수집

유방암 데이터는 Kaggle에서 제공되며 다음 링크에서 얻을 수 있습니다:

[Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

데이터를 다운로드하고 분석에 활용합니다.

```
#https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
#https://drive.google.com/file/d/1Ow8SNjYJeDw69wCJngsaQLqmaFC0PrDn/view?usp=drive_link
system("gdown --id 1Ow8SNjYJeDw69wCJngsaQLqmaFC0PrDn")
system("ls",TRUE)

bc<-fread("BreastCancerData.csv",encoding="UTF-8")%>%as_tibble()
bc%>%show()
```

## 3. 데이터 전처리

유방암 데이터를 불러온 후, 결측값 처리, 이상값 처리 등을 수행합니다. 데이터의 구조를 확인하고 필요한 변수를 팩터(factor)로 변환합니다. 

### 3-1. 이상값 처리
결측값은 없습니다. 정수 데이터 변수의 이상값을 NA처리한 후 삭제하겠습니다.

```
# 이상치 및 결측값 처리 함수
calculate_outliers <- function(data, column_name) {
  iqr_value <- IQR(data[[column_name]])
  upper_limit <- summary(data[[column_name]])[5] + 1.5 * iqr_value
  lower_limit <- summary(data[[column_name]])[2] - 1.5 * iqr_value

  data[[column_name]] <- ifelse(data[[column_name]] < lower_limit | data[[column_name]] > upper_limit, NA, data[[column_name]])

  return(data)
}
table(is.na(bc))
bc_<-select(bc,-diagnosis)#char형 변수를 제외하고 정수형 변수만을 저장한 bc_를 통해 boxplot
boxplot(bc_)
# 이상치 및 결측값 처리 및 결과에 대한 상자그림 그리기
bc <- calculate_outliers(bc, "radius_mean")
bc <- calculate_outliers(bc, "texture_mean")
bc <- calculate_outliers(bc, "perimeter_mean")
bc <- calculate_outliers(bc, "area_mean")
bc <- calculate_outliers(bc, "smoothness_mean")
bc <- calculate_outliers(bc, "compactness_mean")
bc <- calculate_outliers(bc, "concavity_mean")
bc <- calculate_outliers(bc, "concave points_mean")
bc <- calculate_outliers(bc, "symmetry_mean")
bc <- calculate_outliers(bc, "fractal_dimension_mean")
bc <- calculate_outliers(bc, "radius_se")
bc <- calculate_outliers(bc, "texture_se")
bc <- calculate_outliers(bc, "perimeter_se")
bc <- calculate_outliers(bc, "area_se")
bc <- calculate_outliers(bc, "smoothness_se")
bc <- calculate_outliers(bc, "compactness_se")
bc <- calculate_outliers(bc, "concavity_se")
bc <- calculate_outliers(bc, "concave points_se")
bc <- calculate_outliers(bc, "symmetry_se")
bc <- calculate_outliers(bc, "fractal_dimension_se")
bc <- calculate_outliers(bc, "texture_worst")
bc <- calculate_outliers(bc, "perimeter_worst")
bc <- calculate_outliers(bc, "area_worst")
bc <- calculate_outliers(bc, "smoothness_worst")
bc <- calculate_outliers(bc, "compactness_worst")
bc <- calculate_outliers(bc, "concavity_worst")
bc <- calculate_outliers(bc, "concave points_worst")
bc <- calculate_outliers(bc, "symmetry_worst")
bc <- calculate_outliers(bc, "fractal_dimension_worst")


table(is.na(bc))
bc <- na.omit(bc)
table(is.na(bc))
bc_<-select(bc,-diagnosis)#char형 변수를 제외하고 정수형 변수만을 저장한 bc_를 통해 boxplot
#을 그려보겠습니다.
boxplot(bc_)
```

```
FALSE 
17639 

FALSE  TRUE 
17048   591 

FALSE 
12338 
```
<img src="https://github.com/auspicious0/BreastCancer/assets/108572025/74de1ce1-26ef-4e04-9e7a-5e51c7ad5b09.png" width="400" height="400"/>
<img src="https://github.com/auspicious0/BreastCancer/assets/108572025/4e5dc4e0-e218-4018-9609-751adbd06f34.png" width="400" height="400"/>

### 3-2. 데이터 변수의 형 변환 및 삭제
진단(diagnosis) 데이터를 살펴보면 chr형 변수입니다. 하지만 M(악성),F(양성) 두개의 값만을 갖는 것을 확인할 수 있습니다. 따라서 해당 변수를 Factor 형 변수로 변환하겠습니다. 또 id(참여자 주민번호) 변수와 V33(결측값으로 이루어진 변수) 변수를 삭제하겠씁니다.

```
str(bc)
bc%>%summary()
```

```
tibble [569 × 33] (S3: tbl_df/tbl/data.frame)
 $ id                     : int [1:569] 842302 842517 84300903 84348301 84358402 843786 844359 84458202 844981 84501001 ...
 $ diagnosis              : chr [1:569] "M" "M" "M" "M" ...
 $ radius_mean            : num [1:569] 18 20.6 19.7 11.4 20.3 ...
 $ texture_mean           : num [1:569] 10.4 17.8 21.2 20.4 14.3 ...
 $ perimeter_mean         : num [1:569] 122.8 132.9 130 77.6 135.1 ...
 $ area_mean              : num [1:569] 1001 1326 1203 386 1297 ...
 $ smoothness_mean        : num [1:569] 0.1184 0.0847 0.1096 0.1425 0.1003 ...
 $ compactness_mean       : num [1:569] 0.2776 0.0786 0.1599 0.2839 0.1328 ...
 $ concavity_mean         : num [1:569] 0.3001 0.0869 0.1974 0.2414 0.198 ...
 $ concave points_mean    : num [1:569] 0.1471 0.0702 0.1279 0.1052 0.1043 ...
 $ symmetry_mean          : num [1:569] 0.242 0.181 0.207 0.26 0.181 ...
 $ fractal_dimension_mean : num [1:569] 0.0787 0.0567 0.06 0.0974 0.0588 ...
 $ radius_se              : num [1:569] 1.095 0.543 0.746 0.496 0.757 ...
 $ texture_se             : num [1:569] 0.905 0.734 0.787 1.156 0.781 ...
 $ perimeter_se           : num [1:569] 8.59 3.4 4.58 3.44 5.44 ...
 $ area_se                : num [1:569] 153.4 74.1 94 27.2 94.4 ...
 $ smoothness_se          : num [1:569] 0.0064 0.00522 0.00615 0.00911 0.01149 ...
 $ compactness_se         : num [1:569] 0.049 0.0131 0.0401 0.0746 0.0246 ...
 $ concavity_se           : num [1:569] 0.0537 0.0186 0.0383 0.0566 0.0569 ...
 $ concave points_se      : num [1:569] 0.0159 0.0134 0.0206 0.0187 0.0188 ...
 $ symmetry_se            : num [1:569] 0.03 0.0139 0.0225 0.0596 0.0176 ...
 $ fractal_dimension_se   : num [1:569] 0.00619 0.00353 0.00457 0.00921 0.00511 ...
 $ radius_worst           : num [1:569] 25.4 25 23.6 14.9 22.5 ...
 $ texture_worst          : num [1:569] 17.3 23.4 25.5 26.5 16.7 ...
 $ perimeter_worst        : num [1:569] 184.6 158.8 152.5 98.9 152.2 ...
 $ area_worst             : num [1:569] 2019 1956 1709 568 1575 ...
 $ smoothness_worst       : num [1:569] 0.162 0.124 0.144 0.21 0.137 ...
 $ compactness_worst      : num [1:569] 0.666 0.187 0.424 0.866 0.205 ...
 $ concavity_worst        : num [1:569] 0.712 0.242 0.45 0.687 0.4 ...
 $ concave points_worst   : num [1:569] 0.265 0.186 0.243 0.258 0.163 ...
 $ symmetry_worst         : num [1:569] 0.46 0.275 0.361 0.664 0.236 ...
 $ fractal_dimension_worst: num [1:569] 0.1189 0.089 0.0876 0.173 0.0768 ...
 $ V33                    : logi [1:569] NA NA NA NA NA NA ...
 - attr(*, ".internal.selfref")=<externalptr> 
       id             diagnosis          radius_mean      texture_mean  
 Min.   :     8670   Length:569         Min.   : 6.981   Min.   : 9.71  
 1st Qu.:   869218   Class :character   1st Qu.:11.700   1st Qu.:16.17  
 Median :   906024   Mode  :character   Median :13.370   Median :18.84  
 Mean   : 30371831                      Mean   :14.127   Mean   :19.29  
 3rd Qu.:  8813129                      3rd Qu.:15.780   3rd Qu.:21.80  
 Max.   :911320502                      Max.   :28.110   Max.   :39.28  
 perimeter_mean     area_mean      smoothness_mean   compactness_mean 
 Min.   : 43.79   Min.   : 143.5   Min.   :0.05263   Min.   :0.01938  
 1st Qu.: 75.17   1st Qu.: 420.3   1st Qu.:0.08637   1st Qu.:0.06492  
 Median : 86.24   Median : 551.1   Median :0.09587   Median :0.09263  
 Mean   : 91.97   Mean   : 654.9   Mean   :0.09636   Mean   :0.10434  
 3rd Qu.:104.10   3rd Qu.: 782.7   3rd Qu.:0.10530   3rd Qu.:0.13040  
 Max.   :188.50   Max.   :2501.0   Max.   :0.16340   Max.   :0.34540  
 concavity_mean    concave points_mean symmetry_mean    fractal_dimension_mean
 Min.   :0.00000   Min.   :0.00000     Min.   :0.1060   Min.   :0.04996       
 1st Qu.:0.02956   1st Qu.:0.02031     1st Qu.:0.1619   1st Qu.:0.05770       
 Median :0.06154   Median :0.03350     Median :0.1792   Median :0.06154       
 Mean   :0.08880   Mean   :0.04892     Mean   :0.1812   Mean   :0.06280       
 3rd Qu.:0.13070   3rd Qu.:0.07400     3rd Qu.:0.1957   3rd Qu.:0.06612       
 Max.   :0.42680   Max.   :0.20120     Max.   :0.3040   Max.   :0.09744       
   radius_se        texture_se      perimeter_se       area_se       
 Min.   :0.1115   Min.   :0.3602   Min.   : 0.757   Min.   :  6.802  
 1st Qu.:0.2324   1st Qu.:0.8339   1st Qu.: 1.606   1st Qu.: 17.850  
 Median :0.3242   Median :1.1080   Median : 2.287   Median : 24.530  
 Mean   :0.4052   Mean   :1.2169   Mean   : 2.866   Mean   : 40.337  
 3rd Qu.:0.4789   3rd Qu.:1.4740   3rd Qu.: 3.357   3rd Qu.: 45.190  
 Max.   :2.8730   Max.   :4.8850   Max.   :21.980   Max.   :542.200  
 smoothness_se      compactness_se      concavity_se     concave points_se 
 Min.   :0.001713   Min.   :0.002252   Min.   :0.00000   Min.   :0.000000  
 1st Qu.:0.005169   1st Qu.:0.013080   1st Qu.:0.01509   1st Qu.:0.007638  
 Median :0.006380   Median :0.020450   Median :0.02589   Median :0.010930  
 Mean   :0.007041   Mean   :0.025478   Mean   :0.03189   Mean   :0.011796  
 3rd Qu.:0.008146   3rd Qu.:0.032450   3rd Qu.:0.04205   3rd Qu.:0.014710  
 Max.   :0.031130   Max.   :0.135400   Max.   :0.39600   Max.   :0.052790  
  symmetry_se       fractal_dimension_se  radius_worst   texture_worst  
 Min.   :0.007882   Min.   :0.0008948    Min.   : 7.93   Min.   :12.02  
 1st Qu.:0.015160   1st Qu.:0.0022480    1st Qu.:13.01   1st Qu.:21.08  
 Median :0.018730   Median :0.0031870    Median :14.97   Median :25.41  
 Mean   :0.020542   Mean   :0.0037949    Mean   :16.27   Mean   :25.68  
 3rd Qu.:0.023480   3rd Qu.:0.0045580    3rd Qu.:18.79   3rd Qu.:29.72  
 Max.   :0.078950   Max.   :0.0298400    Max.   :36.04   Max.   :49.54  
 perimeter_worst    area_worst     smoothness_worst  compactness_worst
 Min.   : 50.41   Min.   : 185.2   Min.   :0.07117   Min.   :0.02729  
 1st Qu.: 84.11   1st Qu.: 515.3   1st Qu.:0.11660   1st Qu.:0.14720  
 Median : 97.66   Median : 686.5   Median :0.13130   Median :0.21190  
 Mean   :107.26   Mean   : 880.6   Mean   :0.13237   Mean   :0.25427  
 3rd Qu.:125.40   3rd Qu.:1084.0   3rd Qu.:0.14600   3rd Qu.:0.33910  
 Max.   :251.20   Max.   :4254.0   Max.   :0.22260   Max.   :1.05800  
 concavity_worst  concave points_worst symmetry_worst   fractal_dimension_worst
 Min.   :0.0000   Min.   :0.00000      Min.   :0.1565   Min.   :0.05504        
 1st Qu.:0.1145   1st Qu.:0.06493      1st Qu.:0.2504   1st Qu.:0.07146        
 Median :0.2267   Median :0.09993      Median :0.2822   Median :0.08004        
 Mean   :0.2722   Mean   :0.11461      Mean   :0.2901   Mean   :0.08395        
 3rd Qu.:0.3829   3rd Qu.:0.16140      3rd Qu.:0.3179   3rd Qu.:0.09208        
 Max.   :1.2520   Max.   :0.29100      Max.   :0.6638   Max.   :0.20750        
   V33         
 Mode:logical  
 NA's:569      
```

```
bc$diagnosis %>% unique()
```
```
M ` B

```
```
bc <- select(bc,-id,-V33) %>%
      mutate_at("diagnosis",factor)
bc %>% str()
```

```
tibble [569 × 31] (S3: tbl_df/tbl/data.frame)
 $ diagnosis              : Factor w/ 2 levels "B","M": 2 2 2 2 2 2 2 2 2 2 ...
 $ radius_mean            : num [1:569] 18 20.6 19.7 11.4 20.3 ...
 $ texture_mean           : num [1:569] 10.4 17.8 21.2 20.4 14.3 ...
 $ perimeter_mean         : num [1:569] 122.8 132.9 130 77.6 135.1 ...
 $ area_mean              : num [1:569] 1001 1326 1203 386 1297 ...
 $ smoothness_mean        : num [1:569] 0.1184 0.0847 0.1096 0.1425 0.1003 ...
 $ compactness_mean       : num [1:569] 0.2776 0.0786 0.1599 0.2839 0.1328 ...
 $ concavity_mean         : num [1:569] 0.3001 0.0869 0.1974 0.2414 0.198 ...
 $ concave points_mean    : num [1:569] 0.1471 0.0702 0.1279 0.1052 0.1043 ...
 $ symmetry_mean          : num [1:569] 0.242 0.181 0.207 0.26 0.181 ...
 $ fractal_dimension_mean : num [1:569] 0.0787 0.0567 0.06 0.0974 0.0588 ...
 $ radius_se              : num [1:569] 1.095 0.543 0.746 0.496 0.757 ...
 $ texture_se             : num [1:569] 0.905 0.734 0.787 1.156 0.781 ...
 $ perimeter_se           : num [1:569] 8.59 3.4 4.58 3.44 5.44 ...
 $ area_se                : num [1:569] 153.4 74.1 94 27.2 94.4 ...
 $ smoothness_se          : num [1:569] 0.0064 0.00522 0.00615 0.00911 0.01149 ...
 $ compactness_se         : num [1:569] 0.049 0.0131 0.0401 0.0746 0.0246 ...
 $ concavity_se           : num [1:569] 0.0537 0.0186 0.0383 0.0566 0.0569 ...
 $ concave points_se      : num [1:569] 0.0159 0.0134 0.0206 0.0187 0.0188 ...
 $ symmetry_se            : num [1:569] 0.03 0.0139 0.0225 0.0596 0.0176 ...
 $ fractal_dimension_se   : num [1:569] 0.00619 0.00353 0.00457 0.00921 0.00511 ...
 $ radius_worst           : num [1:569] 25.4 25 23.6 14.9 22.5 ...
 $ texture_worst          : num [1:569] 17.3 23.4 25.5 26.5 16.7 ...
 $ perimeter_worst        : num [1:569] 184.6 158.8 152.5 98.9 152.2 ...
 $ area_worst             : num [1:569] 2019 1956 1709 568 1575 ...
 $ smoothness_worst       : num [1:569] 0.162 0.124 0.144 0.21 0.137 ...
 $ compactness_worst      : num [1:569] 0.666 0.187 0.424 0.866 0.205 ...
 $ concavity_worst        : num [1:569] 0.712 0.242 0.45 0.687 0.4 ...
 $ concave points_worst   : num [1:569] 0.265 0.186 0.243 0.258 0.163 ...
 $ symmetry_worst         : num [1:569] 0.46 0.275 0.361 0.664 0.236 ...
```

## 4. 의사 결정 트리 분석(Decision Tree)

데이터를 학습 및 테스트 세트로 분할하고 모델을 생성하여 성능을 평가하고 과적합을 방지하기 위해 가지를 제거합니다.

### 4-1. 데이터 분할

Decision Tree 분석 rpart()를 수행하기 위해 우선 train 데이터와 test 데이터로 데이터를 분리합니다. (데이터가 충분하지않으므로 9:1으로 분리합니다.)
그런데 무작위로 데이터를 분리하지 않고 diagnosis 를 기준으로 데이터를 분할하기 위해 caret을 install해 createDataPartition를 활용하겠습니다.

```
install.packages("caret")
library(caret)
set.seed(31) #다음번 계산 때도 동일한 값으로 분할될 수 있도록 조치

index <- caret::createDataPartition(y = bc$diagnosis, p = 0.9, list=FALSE)
train <- bc[index,]
test <- bc[-index,]
```

### 4-2. 의사 결정 트리 모델 학습

이제 train 데이터를 가지고 DecisionTree 모델을 학습하겠습니다. 

```
install.packages("rpart")
library(rpart)

model_bc <- rpart(formula = diagnosis ~ ., data= train, method = "class")
summary(model_bc)

```

```
Call:
rpart(formula = diagnosis ~ ., data = train, method = "class")
  n= 359 

          CP nsplit rel error    xerror       xstd
1 0.67415730      0 1.0000000 1.0000000 0.09192627
2 0.06741573      1 0.3258427 0.5056180 0.07049103
3 0.06179775      2 0.2584270 0.4606742 0.06771241
4 0.01000000      4 0.1348315 0.3146067 0.05708944

Variable importance
        radius_worst           area_worst      perimeter_worst 
                  16                   15                   13 
           area_mean          radius_mean       perimeter_mean 
                  13                   13                   12 
        texture_mean        texture_worst concave points_worst 
                   3                    3                    3 
      concavity_mean           texture_se  concave points_mean 
                   1                    1                    1 
   compactness_worst      concavity_worst     compactness_mean 
                   1                    1                    1 
             area_se        smoothness_se 
                   1                    1 

Node number 1: 359 observations,    complexity param=0.6741573
  predicted class=B  expected loss=0.2479109  P(node) =1
    class counts:   270    89
   probabilities: 0.752 0.248 
  left son=2 (279 obs) right son=3 (80 obs)
  Primary splits:
      radius_worst         < 16.805    to the left,  improve=80.95968, (0 missing)
      perimeter_worst      < 105.95    to the left,  improve=79.58408, (0 missing)
      area_worst           < 865.7     to the left,  improve=79.26065, (0 missing)
      concave points_mean  < 0.051455  to the left,  improve=76.84859, (0 missing)
      concave points_worst < 0.1416    to the left,  improve=72.36913, (0 missing)
  Surrogate splits:
      area_worst      < 865.7     to the left,  agree=0.992, adj=0.963, (0 split)
      perimeter_worst < 109.75    to the left,  agree=0.969, adj=0.863, (0 split)
      area_mean       < 696.05    to the left,  agree=0.964, adj=0.837, (0 split)
      radius_mean     < 15.045    to the left,  agree=0.961, adj=0.825, (0 split)
      perimeter_mean  < 96.42     to the left,  agree=0.955, adj=0.800, (0 split)

Node number 2: 279 observations,    complexity param=0.06179775
  predicted class=B  expected loss=0.06810036  P(node) =0.7771588
    class counts:   260    19
   probabilities: 0.932 0.068 
  left son=4 (251 obs) right son=5 (28 obs)
  Primary splits:
      concave points_worst < 0.1349    to the left,  improve=13.611100, (0 missing)
      concave points_mean  < 0.048785  to the left,  improve=12.540700, (0 missing)
      concavity_mean       < 0.093405  to the left,  improve=10.696770, (0 missing)
      perimeter_worst      < 101.65    to the left,  improve=10.011740, (0 missing)
      area_worst           < 727.1     to the left,  improve= 9.007121, (0 missing)
  Surrogate splits:
      concave points_mean < 0.04804   to the left,  agree=0.943, adj=0.429, (0 split)
      concavity_mean      < 0.093405  to the left,  agree=0.935, adj=0.357, (0 split)
      compactness_worst   < 0.361     to the left,  agree=0.935, adj=0.357, (0 split)
      concavity_worst     < 0.3373    to the left,  agree=0.932, adj=0.321, (0 split)
      compactness_mean    < 0.1332    to the left,  agree=0.921, adj=0.214, (0 split)

Node number 3: 80 observations,    complexity param=0.06741573
  predicted class=M  expected loss=0.125  P(node) =0.2228412
    class counts:    10    70
   probabilities: 0.125 0.875 
  left son=6 (10 obs) right son=7 (70 obs)
  Primary splits:
      texture_worst   < 19.91     to the left,  improve=10.414290, (0 missing)
      texture_mean    < 16.37     to the left,  improve= 9.252306, (0 missing)
      concavity_mean  < 0.07265   to the left,  improve= 7.023810, (0 missing)
      concavity_worst < 0.2314    to the left,  improve= 6.156410, (0 missing)
      texture_se      < 0.4923    to the left,  improve= 5.327789, (0 missing)
  Surrogate splits:
      texture_mean  < 16.37     to the left,  agree=0.988, adj=0.9, (0 split)
      texture_se    < 0.47315   to the left,  agree=0.912, adj=0.3, (0 split)
      symmetry_mean < 0.14035   to the left,  agree=0.900, adj=0.2, (0 split)
      radius_se     < 0.2171    to the left,  agree=0.900, adj=0.2, (0 split)
      perimeter_se  < 1.5295    to the left,  agree=0.900, adj=0.2, (0 split)

Node number 4: 251 observations
  predicted class=B  expected loss=0.01593625  P(node) =0.6991643
    class counts:   247     4
   probabilities: 0.984 0.016 

Node number 5: 28 observations,    complexity param=0.06179775
  predicted class=M  expected loss=0.4642857  P(node) =0.07799443
    class counts:    13    15
   probabilities: 0.464 0.536 
  left son=10 (17 obs) right son=11 (11 obs)
  Primary splits:
      texture_mean        < 19.45     to the left,  improve=7.810924, (0 missing)
      texture_worst       < 27.49     to the left,  improve=6.706349, (0 missing)
      area_worst          < 724.05    to the left,  improve=5.357143, (0 missing)
      concave points_mean < 0.04944   to the left,  improve=3.778571, (0 missing)
      smoothness_se       < 0.005792  to the left,  improve=3.500000, (0 missing)
  Surrogate splits:
      texture_worst  < 27.49     to the left,  agree=0.893, adj=0.727, (0 split)
      texture_se     < 1.452     to the left,  agree=0.786, adj=0.455, (0 split)
      concavity_mean < 0.10298   to the left,  agree=0.750, adj=0.364, (0 split)
      area_se        < 24.89     to the left,  agree=0.750, adj=0.364, (0 split)
      smoothness_se  < 0.0073995 to the left,  agree=0.750, adj=0.364, (0 split)

Node number 6: 10 observations
  predicted class=B  expected loss=0.2  P(node) =0.02785515
    class counts:     8     2
   probabilities: 0.800 0.200 

Node number 7: 70 observations
  predicted class=M  expected loss=0.02857143  P(node) =0.1949861
    class counts:     2    68
   probabilities: 0.029 0.971 

Node number 10: 17 observations
  predicted class=B  expected loss=0.2352941  P(node) =0.04735376
    class counts:    13     4
   probabilities: 0.765 0.235 

Node number 11: 11 observations
  predicted class=M  expected loss=0  P(node) =0.03064067
    class counts:     0    11
   probabilities: 0.000 1.000 
```

CP 0.01로 더이상 분기하지 않습니다. 그 지점의 오류율(rel_error)과 교차검증오류율(xerror),교차검증오류의 표준편차(xstd)의 값을 확인합니다.

이 지점은 가지치기(pruning) 을 위한 최적의 lowest level 선택에 사용됩니다.

Variable importance 값은 둘레(perimeter_worst)이 가장 크고 반경(radius_worst), 지역(area_worst)이 그 다음을 차지합니다.

그 다음으로 모델에 관한 설명이 나오는데 이는 그림을 통해 살펴보겠습니다.

### 4-3. 모델 시각화

모델을 시각화 해 직관적으로 이해해 보겠습니다.

```
par(mfrow = c(1,1), xpd = NA)
plot(model_bc)
text(model_bc, use.n = TRUE)
```


<img src="https://github.com/auspicious0/BreastCancer/assets/108572025/58e3afb0-b4de-459a-8df2-64a1452aeb29.png" width="400" height="400"/>

위 그림은 식별이 어렵습니다. 따라서 더 식별이 편한 그림으로 바꿔 보겠습니다.

```
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(model_bc)

install.packages(c("rattle","rpart.plot"))

library(rattle)
library(rpart.plot)
library(RColorBrewer)

fancyRpartPlot(model_bc)
```

<img src="https://github.com/auspicious0/BreastCancer/assets/108572025/b43809e6-36ac-4b7f-a178-dd7574e3d6f0.png" width="400" height="400"/>
<img src="https://github.com/auspicious0/BreastCancer/assets/108572025/78a9cb98-7bbf-4e79-8232-18b30d9a3bbe.png" width="400" height="400"/>


radius_worst, area_worst, perimeter_worst 순으로 Variable importance를 차지하고 있으나

1. 데이터 양이 많지 않고
 
2. Variable importance 값 차이가 크지 않기 때문에 예상과는 다른 결정트리가 나왔습니다.

해당 트리가 과잉적합에 빠지지 않도록 모델 model_bc에 가지치기(pruning)을 하려고 합니다.


### 4-4. 가지치기(pruning)

우선, 교차 검증 오류율(xerror)이 최소가 되는 CP를 min_xerror_cp에 저장하게습니다

```
min_xerror_cp <- model_bc$cptable %>%
  as_tibble() %>%
  filter(xerror == min(xerror)) %>%
  pull(CP)
print("min_error_cp = ")
min_xerror_cp
```

```
[1] "min_error_cp = "
0.01
```

위에서 구한 min_xerror_cp 값을 이용하여 모델에 가지치기(pruning)를 수행하여 model_pr에 저장하겠습니다.

```
model_pr <- rpart::prune(model_bc, cp = min_xerror_cp)
fancyRpartPlot(model_pr)
```
![image](https://github.com/auspicious0/BreastCancer/assets/108572025/4e6b362c-ceed-45f7-895d-c2938a1dd00c)

## 5. 모델 평가

### 5-1. 예측

```
predict_value <- predict(model_pr, test, type = "class") %>% tibble(predict_value = .)
predict_check <- test %>% select(diagnosis) %>% dplyr::bind_cols(., predict_value)
predict_check
```

```

# A tibble: 39 × 2
   diagnosis predict_value
   <fct>     <fct>        
 1 M         M            
 2 B         B            
 3 B         B            
 4 M         M            
 5 B         B            
 6 B         B            
 7 B         B            
 8 M         M            
 9 M         M            
10 B         B            
# ℹ 29 more rows
```
모두 예측을 수행한 것을 확인할 수 있습니다.혼돈 메트릭스(confusionMatrix)를 활용하여 모델을 분석해 보겠습니다.

### 5-2. 혼돈 메트릭스(Confusion Matrix)

```
cm <- caret::confusionMatrix(predict_check$predict_value, test$diagnosis)
cm
```

```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 30  0
         M  0  9
                                     
               Accuracy : 1          
                 95% CI : (0.9097, 1)
    No Information Rate : 0.7692     
    P-Value [Acc > NIR] : 3.599e-05  
                                     
                  Kappa : 1          
                                     
 Mcnemar's Test P-Value : NA         
                                     
            Sensitivity : 1.0000     
            Specificity : 1.0000     
         Pos Pred Value : 1.0000     
         Neg Pred Value : 1.0000     
             Prevalence : 0.7692     
         Detection Rate : 0.7692     
   Detection Prevalence : 0.7692     
      Balanced Accuracy : 1.0000     
                                     
       'Positive' Class : B      
```
정확도 (Accuracy): 1.0

전체 예측 중에서 올바르게 분류한 비율로, 1.0또는 100%입니다.(TP + TN) / (TP + TN + FP + FN)

민감도 (Sensitivity): 1.0

실제 양성 중에서 올바르게 양성으로 분류된 비율로, 1.0또는 100%입니다.

특이도 (Specificity): 1.0

실제 음성 중에서 올바르게 음성으로 분류된 비율로, 1.0또는 100%입니다.

정밀도 (Precision): 1.0

정밀도는 모델이 양성으로 예측한 샘플 중에서 실제로 양성인 샘플의 비율을 나타냅니다. TP / (TP + FP)

재현율 (Recall): 1.0

재현율은 실제로 양성인 샘플 중에서 모델이 양성으로 예측한 샘플의 비율을 나타냅니다. TP / (TP + FN)

데이터가 작기 때문에 가능한 결과라고 생각하지만 100프로의 정확도 및 정밀도 등을 보인다는 점이 의미가 있고 결정트리를 통해 분석하기 좋은 데이터다라는 결론을 내립니다.

## 6. 문의
프로젝트에 관한 문의나 버그 리포트는 [이슈 페이지](https://github.com/auspicious0/BreastCancer/issues)를 통해 제출해주세요.

보다 더 자세한 내용을 원하신다면 [보고서](https://github.com/auspicious0/BreastCancer/blob/main/DesicionTree.ipynb) 를 확인해 주시기 바랍니다.
