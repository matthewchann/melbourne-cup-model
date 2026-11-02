#A simple example of the technique behind our Melbourne Cup model, on made-up races.
#None of our real factors, weights or data are here. The terms used are explained at the bottom.

set.seed(2026)                                                    #makes the random numbers the same every run


#==================================================================
#1 Each horse's chance of winning = its share of the race
#==================================================================

#P(horse i wins) = exp(score_i) / sum of exp(score) over every horse in the same race

race_share <- function(score, race) {

  e <- exp(score - ave(score, race, FUN = max))                  #exp(score), after subtracting the race's top score
                                                                  #(stops huge numbers; the shares don't change)

  e / ave(e, race, FUN = sum)                                     #divide by the race total, so each race adds up to 1
}


#==================================================================
#2 Make up races where we know the true answer
#==================================================================

make_races <- function(n) {

  field <- sample(8:16, n, replace = TRUE)                        #a field size of 8 to 16 for each race

  d <- data.frame(race = rep(1:n, field),                         #one row per horse, labelled with its race number
                  x1   = rnorm(sum(field)),                       #made-up factor 1 for every horse
                  x2   = rnorm(sum(field)))                       #made-up factor 2 for every horse

  p <- race_share(0.8 * d$x1 - 0.5 * d$x2, d$race)                #true chances: the true weights are 0.8 and -0.5

  d$win <- FALSE                                                  #nobody has won yet

  for (r in 1:n) {                                                #for each race...

    horses <- which(d$race == r)                                  #...find its horses

    d$win[sample(horses, 1, prob = p[horses])] <- TRUE            #...and draw one winner using the true chances
  }

  d
}

train <- make_races(3000)                                         #3,000 races to fit on

test  <- make_races(1000)                                         #1,000 races the fit never sees


#==================================================================
#3 Fit the weights
#==================================================================

#Find the weights b that give the actual winners the highest chances (maximum likelihood),
#minus a ridge penalty on large weights, with factor 1 kept >= 0 and factor 2 kept <= 0

fit <- function(d, penalty = 1) {

  X <- as.matrix(d[, c("x1", "x2")])                              #the factors as a grid of numbers, one row per horse

  loss <- function(b) {

    chance <- race_share(as.vector(X %*% b), d$race)              #each horse's chance with weights b

    -sum(log(chance[d$win])) + penalty / 2 * sum(b^2)             #minus the log-likelihood of the winners, plus the penalty
  }

  optim(c(0, 0), loss, method = "L-BFGS-B",                       #start both weights at 0 and search for the lowest loss
        lower = c(0, -Inf), upper = c(Inf, 0))$par                #the limits on each weight; $par = the best weights found
}

b <- fit(train)


#==================================================================
#4 Test on races the fit never saw
#==================================================================

chance <- race_share(as.vector(as.matrix(test[, c("x1", "x2")]) %*% b), test$race)   #chances in the test races

model  <- -mean(log(chance[test$win]))                            #log-loss: average -log(chance given to the winner)

random <- -mean(log(1 / table(test$race)))                        #log-loss of a random pick: each horse gets 1 / field size

cat(sprintf("Fitted weights: %.2f, %.2f (true: 0.80, -0.50)\n", b[1], b[2]))

cat(sprintf("Pseudo-R2 on unseen races: %.3f\n", 1 - model / random))   #1 - model log-loss / random log-loss


#==================================================================
#Glossary
#==================================================================

#race_share         Our name for the "share of the race" formula. In statistics it's called the softmax function.
#
#ave(x, race, FUN)  Works out FUN (e.g. max or sum) within each race, and repeats the answer on every horse in it.
#
#sample(...)        Picks at random. sample(8:16, n, replace = TRUE) picks n field sizes; with prob = it weights the pick.
#
#data.frame         A table: one row per horse, with a column for each thing we know about it.
#
#rep(1:n, field)    Repeats each race number once per horse, e.g. rep(1:2, c(3, 2)) gives 1 1 1 2 2.
#
#rnorm(k)           k random numbers from a bell curve centred on 0, used here as made-up factor values.
#
#d$race, d$win      The race and win columns of table d. d$win is TRUE for the winner, so chance[d$win] keeps only the winners' chances.
#
#as.matrix          Turns the factor columns into a plain grid of numbers so they can be multiplied by the weights.
#
#X %*% b            Matrix multiplication: for every horse, factor 1 x weight 1 + factor 2 x weight 2 = its score.
#
#penalty            How strict the ridge penalty is. Cost = penalty / 2 x (weight 1 squared + weight 2 squared), pulling weights towards 0.
#
#optim              R's built-in tool for finding the inputs that make a function as small as possible (here, the loss).
#
#L-BFGS-B           Limited-memory Broyden-Fletcher-Goldfarb-Shanno, with Bounds: optim's search method that allows a lower and upper limit on each weight.
#
#$par               The part of optim's answer that holds the best weights it found.
#
#table(test$race)   Counts the horses in each race (the field sizes).
