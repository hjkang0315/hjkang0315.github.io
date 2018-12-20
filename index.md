---
layout: default
permalink: index.html
title: Personal Homepage of foo boo
description: "Blogging on ...."
---

본 정리 내용은 고려대학교 산업경영공학과 강필성교수님의 Business Analytics 강의를 바탕으로 작성되었습니다.

****Generative Models
크게 보면 Machine Learning은 환경을 모델링하는 과정으로, Data라는 경험을 통해 학습을 진행하고 이를 통해 새로운 상황이 발생했을 때 추론을 하는 과정이라 할 수 있다. 이를 3가지의 큰 범주로 나누면 다음과 같다.
(1) Generative model : Data가 어떤 확률분포로부터 생성되었는지 학습, Joint probability를 최대화하는 것이 목적, ex) Hidden Markov Model, Generative Adversarial Network 등. 상대적인 장점으로, 비교적 데이터가 적어도 진행이 가능하고, 주어진 데이터를 통해 데이터의 특성을 파악할 수 있으며, 제대로 된 특성을 찾았다면 해당 분포를 따르는 데이터를 생성해낼 수 있음
(2) Discriminative model : 주어진 x(=v)로 y(=h)를 예측하는 일반적인 Machin learning 종류, Conditional probability를 최대화하는 것이 목적
(3) Undirected model : x(=v)와 y(=h)의 방향없이 가장 낮은 Energy 상태를 만드는 Joint Probability를 계산, ex) Restricted Boltzmann Machine


















References
Fox-Roberts, P., & Rosten, E. (2014). Unbiased generative semi-supervised learning. The Journal of Machine Learning Research, 15(1), 367-443.
Kingma, D. P., Mohamed, S., Rezende, D. J., & Welling, M. (2014). Semi-supervised learning with deep generative models. In Advances in Neural Information Processing Systems (pp. 3581-3589).
Zhu, X. (2007). Semi-Supervised Learning Tutorial. International Conference on Machine Learning (ICML 2007).
Choi, S. (2015). Deep Learning: A Quick Overview. Deep Learning Workship. KIISE.

