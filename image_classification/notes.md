http://vision.stanford.edu/teaching/cs231n-demos/knn/



loss function
------------------------------------------------------------------------------------------
softmax    :  L = -(1/N)∑i∑jI(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)

hinge loss : L = (1/N)∑i∑j#yi max(0, fj-fyi+1)
------------------------------------------------------------------------------------------



grad of loss:
------------------------------------------------------------------------------------------
softmax-grad : ∇Wk(L) = -(1/N)∑i xiT (I(k==yi)-Pk) + 2λWk, where Pk = exp(fk)/∑jexp(fj)

hinge-grad   :
------------------------------------------------------------------------------------------


accuracy of some strategy:

random-init weights : 10%            | limit-acc : 15%


challenges of recognition
1. illumination           | 光照变化
2. deformation            | 形态变化
3. occlusion              | 遮挡
4. clutter                | 前后景交融
5. intraclass variation   | 类内变化
