본 포스트는 고려대학교 산업경영공학과 강필성교수님의 Business Analytics 강의를 바탕으로 작성되었습니다.

# Generative Models
크게 보면 Machine Learning은 환경을 모델링하는 과정으로, Data라는 경험을 통해 학습을 진행하고 이를 통해 새로운 상황이 발생했을 때 추론을 하는 과정이라 할 수 있다. 이를 3가지의 큰 범주로 나누면 다음과 같다.
1. Generative model : Data가 어떤 확률분포로부터 생성되었는지 학습, Joint probability를 최대화하는 것이 목적, ex) Hidden Markov Model, Generative Adversarial Network 등. 상대적인 장점으로, 비교적 데이터가 적어도 진행이 가능하고, 주어진 데이터를 통해 데이터의 특성을 파악할 수 있으며, 제대로 된 특성을 찾았다면 해당 분포를 따르는 데이터를 생성해낼 수 있음
2. Discriminative model : 주어진 x(=v)로 y(=h)를 예측하는 일반적인 Machin learning 종류, Conditional probability를 최대화하는 것이 목적
3. Undirected model : x(=v)와 y(=h)의 방향없이 가장 낮은 Energy 상태를 만드는 Joint Probability를 계산, ex) Restricted Boltzmann Machine

![이미지1](http://hjkang0315.github.io/1.png)

## Semi-supervised learning에서의 Gaussian mixture model기반 Generative model
본 포스트에서 Semi-supervised learning을 진행함에 있어 Gaussian mixture model에 기반한 Generative model을 보인다. 수업에서 Generative model로서 다룬 내용은 일반적인 Gaussian mixture model과 전체적인 개념과 철학은 같지만 Process의 차이가 있다.
1. Novelty detection관점에서 Gaussian mixture model은 주어진 모든 data는 정상이라는 가정을 하고 밀도를 추정하는 것 (Mode 수만 지정, 초기 θ Random)
2. Semi-supervised learning관점의 분포 기반 Generative model은 각각의 소수의 Label class에 따라서 서로 다른 분포를 가정을 하고, 그후 Unlabeled data를 고려해서 추정을 통해 Boundary를 찾는 것. 알고리즘 진행상의 순서로 차이를 요약하면, m=2인 Binary Class의 경우를 가정, 소수의 Labeled Data만을 가지고 추정 값인 θ=(ω_1, ω_2, μ_1, μ_2, Σ_1, Σ_2)의 초기값을 잡고, 이후 Unlabeled data를 추가하여 EM step을 진행하는 부분이 차이점

## Gaussian mixture model기반 Generative model
개별 Class는 각 하나의 Single Gaussian 분포로부터 비롯되었다고 가정 한다. 아래 그림을 예로 설명하면, (o) Class와, (+) Class가 각각의 Single Gaussian 분포를 따른다고 가정하자.

![이미지2](http://hjkang0315.github.io/2.png)

먼저 주어진 일부 Data는 (o) Class와, (+) Class로 Labeled되어있기 때문에, Class별로 Mean vector μ를 구할 수 있고 그에 대한 Covariance matrix Σ를 구할 수 있다. 단순한 계산으로 θ를 얻을 수 있지만, 굳이 따져보면 Labeled Data로부터 Maximization Step을 진행하는 것이다. 해당하는 Mode에서부터 확률분포를 추정하면, (o) Class와, (+) Class의 생성 확률이 같아지는 특정한 점들을 찾을 수 있고, 해당 점들을 통해서 경계를 구분한다. 여기서, ω_1과 ω_2는 각 Class의 비중(Weight)이며, μ_1과 μ_2는 각 Class의 Mean vector, Σ_1과 Σ_2는 각 Class의 Covariance matrix이다. Covariance matrix에 의해서 분포에 따른 등고선으로 표현할 수 있다.

## 많은 수의 Unlabeled Data와 소수의 Labeled Data가 주어진 상황에서, Generative model의 관점에서 접근하면 어떠한 Decision boundary를 찾아낼 수 있을까?
X와 θ가 주어졌을 때 y를 구하는게 우리 목적이다. 따라서, 초기에 주어진 Labeled Data에 의해 추정된 θ값 환경에 대해 새로운 Unlabeled data(X)들이 추가되면 θ_1로부터 Unlabeled data가 생성되었을 확률 값과, θ_2로부터 Unlabeled data가 생성되었을 확률 값, 그리고 Data에 대한 불균형이 있다면 해당 특징을 반영하는 사전 확률(Prior probability)을 고려해서 생성 확률이 더 높은 θ의 Class에 할당(y할당)하는 방법으로 Classification을 진행한다. 이는 먼저 추정된 θ를 통해 조건부 확률을 구하는 Expectation step이다. 일반적인 Classification과 Boundary를 찾고자 하는 목적은 같지만 과정이 다르다.

![이미지3](http://hjkang0315.github.io/3.png)

이어서, Class를 할당한 Labeled Data를 사용해 다시 θ=(ω_1, ω_2, μ_1, μ_2, Σ_1, Σ_2)를 추정하는 Maximization Step을 진행한다. EM Step을 반복하여 최적의 분포를 찾아가는데, 초기 Labeled Data(X_l)를 통해 조건부확률을 구하고 다수의 Unlabeled Data(X_u)를 추가로 고려하여 조건부확률을 구하는 것이므로, 이에 따라 Decision boundary는 달라지게 된다.

![이미지4](http://hjkang0315.github.io/4.png)

앞선 설명을 간단히 요약하면, ①소수의 Labeled Data를 통해 θ추정(M step), ②추정된 θ에 Unlabeled Data를 추가하여 각 Point들에 대해 조건부확률을 구하고(E Step) 확률에 따라 Class(y)를 할당, ③확장된 Labeled Data 공간에서 다시 θ추정(M step) ④ EM Step을 반복하여 최적 결과 도출하는 과정을 거친다.

위에서 설명한 과정을 수식을 통해 설명하면, Quantity of interests는 Maximize를 해야하는 Joint Probability이며, 최대가 되는 θ=(ω_m, μ_m, Σ_m)를 찾아내는 것이 목적이다.
![이미지5](http://hjkang0315.github.io/5.png)

Binary Classification에 적용하면, Labeled Data에 대한 MLE는 다음과 같다. Labeled Data의 경우 θ는 해당하는 class에 대해 ω : Frequency, μ : Sample mean, Σ : Sample covariance를 계산하면 간단히 얻을 수 있다.
![이미지6](http://hjkang0315.github.io/6.png)

u개의 Unlabeled data가 포함되면 다음과 같이 표현할 수 있다. (Labeled data l개, Unlabeled data u개) Unlabeled data를 나타내는 term에서 Class가 Binary로 y∈{1,2}인 경우라면, 두가지 class인 경우의 확률을 따로 계산하여 모두 더한 것으로 표현된다. Unlabeled y가 Hidden variable이므로 MLE를 얻는 것이 보다 어렵다.
![이미지7](http://hjkang0315.github.io/7.png)

순서대로 살펴보면, Step0은 초기 주어진 소수의 Labeled data로 만들어진 최적의 θ로부터 EM알고리즘을 시작한다. Step1은 E step을 진행한다. x∈X_u이고, Class y가 1일 때 해당 x가 생성될 확률, 2일 때 해당 x가 생성될 확률을 계산한다. 예를 들어, p(x,1│θ)=0.02 및 p(x,2│θ)=0.01로 구해진다면, p(1│x_u,θ)=0.02/(0.02+0.01)=0.67 및 p(2│x_u,θ)=0.01/(0.02+0.01)=0.33로 계산되어 해당 unlabeled x가 각 class의 y∈{1,2}에 속할 확률을 나타낸다. 이에 따라 해당 x를 더 높은 확률의 Class에 할당한다.
![이미지8](http://hjkang0315.github.io/8.png)

Step2는 앞선 과정을 통해 Labeled된 x들의 정보를 이용해 M step을 진행하여 MLE θ로 update한다. 이 과정을 반복하여 Solution을 찾는다.
![이미지9](http://hjkang0315.github.io/9.png)

## 일반화된 EM알고리즘
다음은 EM알고리즘의 일반화된 내용이다. 여기서 H는 Unlabeled data이다. θ가 주어졌을 때 data가 생성될 확률은 hidden data의 class는 모르기 때문에, 이에 대해 모든 확률을 따져서 Joint probability를 구하고 data set이 가장 Maximum Likely하게 생성될 수 있는 확률 분포를 찾아내겠다는 의미로 목적은 동일하다.
![이미지10](http://hjkang0315.github.io/10.png)

## 장점과 단점
EM알고리즘을 기본으로 한 Generative model은 명백하고 많이 연구된 확률 Framework이며, Model이 올바르다면 아주 효과적이다. 하지만 반대로 Self-training의 경우와 같이 Model이 잘못되었다면 결과는 더 좋지않은 방향으로 갈 수 있으며, Local optimal의 위험이 있고 Correctness를 확인하기가 어렵다. 예를 들어, 흔치는 않지만 아래의 그래프처럼 위 아래로 Class가 존재하지만 data가 좌우 경향으로 분포된 경우 Optimal을 잘 찾지 못하는 단점을 갖는다.
![이미지11](http://hjkang0315.github.io/11.png)

구현 코드는 다음과 같다.
```Python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class GenerativeModels:
    def __init__(self, labeled_x, labeled_y, num_class, unlabeled_x):
        self.labeled_x = labeled_x
        self.labeled_y = labeled_y
        self.num_class = num_class
        self.unlabeled_x = unlabeled_x
        
    def fit(self, max_iter=10):
        Weight, Mu, Sigma = self.initialize()  ##  label 데이터를 통해 parameter theta인 w, mu, sigma 추정
        
        self.plot_data(Mu, Sigma)
        
        iterator = 0
        previous_mle = -99999
        while(True):
            iterator += 1
            
            '''expectation step : Unlabeled Data를 추가하여 앞서 추정된 theta에 대한
               각 Point들의 조건부확률을 구하고(E Step) 확률에 따라 Class(y)를 할당'''
            m = self.expectation(self.unlabeled_x, Weight, Mu, Sigma)
            
            ''' maximization step : 확장된 Labeled Data 공간에서 다시 theta를 추정'''
            Weight, Mu, Sigma = self.maximization(self.unlabeled_x, m)
            
            if (iterator % 5) == 0:
                self.plot_data(Mu, Sigma)
            
            mle = self.maximum_log_like(self.unlabeled_x, Weight, Mu, Sigma)
            print("loop [", iterator, "] MLE=", mle, " DIFF=", abs(mle-previous_mle))
            
            '''지정된 iterator만큼 e-step과 m-step을 반복하거나,
               더 이상 MLE값이 유의미하게 개선되지 않으면 종료'''
            if (iterator > max_iter or abs(mle-previous_mle) < 0.01):
                break
                
            previous_mle = mle
        
        return Weight, Mu, Sigma

    def plot_data(self, Mu, Sigma):
        nx = np.arange(-2.0, 7.0, 0.1)
        ny = np.arange(-2.0, 7.0, 0.1)
        ax, ay = np.meshgrid(nx, ny)
        
        plt.scatter(self.labeled_x[0:10, 0], self.labeled_x[0:10, 1], c="r", s=3.5)
        plt.scatter(self.labeled_x[10:, 0], self.labeled_x[10:, 1], c="b", s=3.5)
        plt.scatter(self.unlabeled_x[0:,0], self.unlabeled_x[0:,1], c="y", s=1)
        
        az_list = []
        for k in range(0, self.num_class):
            az = mlab.bivariate_normal(ax, ay, Sigma[0, 0, k], Sigma[1, 1, k], Mu[0,k], Mu[1,k], Sigma[1, 0, k])
            az_list.append(az)

        az = 10.0 * (az_list[1] - az_list[0])
        contour = plt.contour(ax, ay, az)
        plt.clabel(contour, inline=0.01, fontsize=10)
        
        plt.show()
    
    def norm_pdf_multivariate(self, x, mu, sigma):
        size = len(x)
        det = np.linalg.det(sigma)
        
        ## 1 / (2*pi^(d/2) * def(sigma)^2)
        norm_const = 1.0/(np.math.pow((2*np.pi), float(size)/2) * np.math.pow(det, 1.0/2))
        x_mu = np.matrix(x - mu)
        inv_ = np.linalg.inv(sigma)
        result = np.math.pow(np.math.e, -0.5 * (x_mu.T * inv_ * x_mu))
        
        return norm_const * result

    def maximum_log_like(self, dataset, Weight, Mu, Sigma):
        K = len(Weight)
        N,M = dataset.shape
        P = np.zeros([N,K])
        
        for k in range(K):
            for i in range(N):
                P[i,k] = self.norm_pdf_multivariate(dataset[i,:][None].T,Mu[0:,k][None].T,Sigma[:,:,k])
                
        return np.sum(np.log(P.dot(Weight)))

    def expectation(self, dataset, Weight, Mu, Sigma):
        N = dataset.shape[0] ## N : the number of instance
        K = len(Weight)      ## K : the number of class
        m = np.zeros([N,K])  ## m : N * K matrix (각 data가 어떤 class에 속할지에 대한 확률)
        for k in range(K):
            for i in range(N):
                m[i,k] = Weight[k]*self.norm_pdf_multivariate(dataset[i,:][None].T, Mu[:,k][None].T,Sigma[:,:,k])
        
        m = m * np.reciprocal(np.sum(m,1)[None].T)
        return m

    def maximization(self, dataset, m):
        N, M = dataset.shape
        K = m.shape[1]
        N_k = np.sum(m,0) ## class별 확률의 합
        Weight = N_k/np.sum(N_k)
        
        ## Mu 계산
        Mu = dataset.T.dot(m).dot(np.diag(np.reciprocal(N_k)))
        
        ## Sigma 계산
        Sigma = np.zeros([M,M,K])
        for k in range(K):
            datMeanSub = dataset.T - Mu[0:,k][None].T.dot(np.ones([1,N]))
            Sigma[:,:,k] = (datMeanSub.dot(np.diag(m[0:,k])).dot(datMeanSub.T))/N_k[k]
        
        return Weight, Mu, Sigma

    ## label 데이터를 통해서 mu, Sigma 계산
    def initialize(self):
        N, M = self.labeled_x.shape
        m = np.zeros([N, self.num_class])

        for i in range(self.labeled_y.shape[0]):
            m[i, np.int(self.labeled_y[i])] = 1            
        
        Weight, Mu, Sigma = self.maximization(self.labeled_x, m)
        
        return Weight, Mu, Sigma
            
			
# 2차원의 다중정규분포를 따르는 난수 생성
mean1=[3,1]
sigma1=[[1,0],[0,1]]
np.random.seed(0)
N1=np.random.multivariate_normal(mean1,sigma1,1000).T

mean2=[2,2]
sigma2=[[1,0.5],[0.5,1]]
np.random.seed(1)
N2=np.random.multivariate_normal(mean2,sigma2,1000).T


# labeled data : 10건, unlabeled data : 990건
labeled_x1 = N1.T[0:10]
labeled_x2 = N2.T[0:10]

unlabeled_x1 = N1.T[10:]
unlabeled_x2 = N2.T[10:]

label_x = np.concatenate((labeled_x1, labeled_x2))
unlabel_x = np.concatenate((unlabeled_x1, unlabeled_x2))

y = np.concatenate((np.zeros(10), np.ones(10)))


gm = GenerativeModels(label_x, y, 2, unlabel_x)


Weight, Mu, Sigma = gm.fit(200)

```


![plot1](http://hjkang0315.github.io/r_1.png)
```Python('loop [', 1, '] MLE=', -6047.941786594476, ' DIFF=', 93951.05821340553)```

![plot2](http://hjkang0315.github.io/r_2.png)
```Python('loop [', 5, '] MLE=', -6020.768392058435, ' DIFF=', 1.899425788508779)```

###...

![plot2](http://hjkang0315.github.io/r_3.png)
```Python('loop [', 60, '] MLE=', -5993.801651950046, ' DIFF=', 0.011293848091554537)```



```Python
Weight
array([0.46829954, 0.53170046])

Mu
array([[2.99390089, 1.99189662],
       [0.89543446, 2.00986136]])
	   
Sigma
array([[[0.97649666, 0.97369508],
        [0.03909626, 0.47806756]],

       [[0.03909626, 0.47806756],
        [0.86571878, 1.01197117]]])
```


###### References
Fox-Roberts, P., & Rosten, E. (2014). Unbiased generative semi-supervised learning. The Journal of Machine Learning Research, 15(1), 367-443.
Kingma, D. P., Mohamed, S., Rezende, D. J., & Welling, M. (2014). Semi-supervised learning with deep generative models. In Advances in Neural Information Processing Systems (pp. 3581-3589).
Zhu, X. (2007). Semi-Supervised Learning Tutorial. International Conference on Machine Learning (ICML 2007).
Choi, S. (2015). Deep Learning: A Quick Overview. Deep Learning Workship. KIISE.