# Predictive-Model-Selection-for-Movies
Using linear regression, MARS, KNN and Neural Network to minimize MSE to find the best model for movies
# Data Source and Goals
We are tasked with working on the ‘Netflix’ dataset. Netflix is an online movie streaming service. Their customers watching movies, and subsequently rate them. Netflix uses these ratings to inform other recommendations. We are going to focus on the first 5 Rocky movies, which is a franchise that features Sylvester Stallone playing an underdog boxer. We will explore the data and predict how highly a consumer will rate Rocky 5 given the ratings of the previous 4 films. We will also have to contend with missing values.
# Modelling Results and Conclusion 
Best Linear Regression: Rocky5 = -0.014320*I(rocky1^2) -0.067756 * I(rocky2^2) -0.070520* I(rocky4^2) -0.006302 *I(rocky1 * rocky2 * rocky3)+ 0.010673*I(rocky1 * rocky2 * rocky4)+ 0.028837*I(rocky2 * rocky3 * rocky4) + 0.784677* log(rocky2)+1.047656 *log(rocky4) + 1.048527

Best MARS model: earth(rocky5~rocky1+rocky2+rocky3+rocky4+log(rock y3)+log(rocky4), data = completeDB,degree = 3)

Best K-Nearest Neighbors model: kknn(allModelsList[[31740]], completeDB,validationdata, k=10, distance = 1)

Best Neural Network model: nnet(rocky5~rocky1+rocky2+rocky3+rocky4, data = completeDB,linout=1,size = 4,maxit = 10000)
