library(ggplot2)
library(tidyverse)
library(latex2exp)
# Asymptotic convex function 

F = function(n,y){
  X = rnorm(n)
  return(y^2+mean(X)*cos(5*y))
}



# plot
X = seq(-1,1,0.01)
ss = c(5,50,100,1000)

par(mfrow = c(3,4))

### generate plots for different n, 3plots for each setting
j=1
set.seed(202103)
repeat{
  for(i in 1:4) {
    plot(X, F(ss[i], X),col="black", type = "l", ylab = '' , xlab = "", 
         xaxt = 'n', yaxt='n', main = paste('fn , n =', ss[i]), cex.main = 2.5 ) }
    j=j+1
  if ( j == 4) {break}
}
