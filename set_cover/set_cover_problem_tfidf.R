#### Some wackiness for the set cover problem ###

# Idea : To determine which sets absolutely need to be in the cover,
# Calculate the tf-idf of each element of each set thus:

# let total number of sets be denoted as N

# Given set I, if element a is in it, tf(a, I) = 1, else tf(a,I) = 0
# IDF : If an element a is in set I, then count its total occurences across all sets
# Call this n_a

#then idf(a, I) = log(N / n_a)
# Therefore tfidf(a, I, N) = 1. log(N/n_a)

# Second approach : use tf=idf but normalize by the number of elements in the set

# Trivial example
EPSILON <- 1.0e-12

library(ggplot2)

# convenient ggplot regression function

ggplotRegression <- function (fit) {
  
  require(ggplot2)
  
  ggplot(fit$model, aes_string(x = names(fit$model)[2], y = names(fit$model)[1])) + 
    geom_point() +
    stat_smooth(method = "lm", col = "red") +
    labs(title = paste("Adj R2 = ",signif(summary(fit)$adj.r.squared, 5),
                       "Intercept =",signif(fit$coef[[1]],5 ),
                       " Slope =",signif(fit$coef[[2]], 5),
                       " P =",signif(summary(fit)$coef[2,4], 5)))
}

# Universe : {1,2,3,4,5,6,7,8,9,10}
n_max <- 100000 # new example

max_set_size <- 500 # for fun

U <- c(1:n_max)

num_sets <- 5000
sets <- list()
num_ele_sets <- list()

for (i in 1:num_sets) {
  
  # how many elements in this set ? choose randomly
  
  num_ele_sets[[i]] <- sample(max_set_size, 1)
  
  sets[[i]] <- sample(n_max, num_ele_sets[[i]], replace = F) # this is consistent in definition to a set as well
  
}

# Now the craziness : calculate tf-idf for each element of each set

# below is not quite idf as we know ; its occurence across all sets o a misnomer
counts <- numeric(length = length(U))
idf <- numeric(length = length(U))

# avoid infinities by making all counts equal to a low EPSILON


for (j in 1:num_sets) {
    v <- unlist(sets[[j]])
    
    counts[v] <- counts[v] + 1
}

for (j in 1:length(counts)) {
  if (counts[j] < EPSILON) {
    counts[j] <- EPSILON
  }
}

# Compute the IDF beforehand as it is a simple version that is set-independent after count
# calculation

for (j in 1:length(counts)) {
  idf[j] <- log(num_sets / counts[j]) # precompute idf to save time
}

#tf <- numeric(length = num_sets)

tf_idf <- numeric(length = num_sets)
normalized_tf_idf <- numeric(length = num_sets)
normalized_counts <- numeric(length = num_sets)

for (j in 1:num_sets) {
  v <- unlist(sets[[j]])
  
  for (val in v) {
# compute the "tf-idf"
    tf_idf[j] <- tf_idf[j] + 1*idf[val]
    
    normalized_counts[j] <- normalized_counts[j] + idf[val]
  }
  
  # now a normalized tf-idf
  normalized_tf_idf[j] <- tf_idf[j] / length(v) # normalization #
  
  # normalized counts
  normalized_counts[j] <- normalized_counts[j] / length(v) # normalization #
  
}

# normalized tf-idf first up ###
#####
#####

ordered_normalized_tf_idf_indexes <- order(normalized_tf_idf, decreasing = T)

normalized_tf_idf_2 <- normalized_tf_idf[ordered_normalized_tf_idf_indexes]

set_seq_normalized_df <- data.frame(set_index = c(1:num_sets),
                         set_seq = ordered_normalized_tf_idf_indexes,
                         normalized_tf_idf_2 = normalized_tf_idf_2
)

# final ordering and stop when all elements covered

constructed_universe <- c()

normalized_set_cover_size_df <- data.frame(
  ctr = numeric(10*length(U)),
  attempted_set = numeric(10*length(U)),
  actual_set = character(10*length(U)),
  size = numeric(10*length(U)),
  stringsAsFactors = F
)

ctr <- 1

for (j in set_seq_normalized_df$set_seq) {
  
  cat("Choice number ", ctr, "\n")
  len_constructed_universe_1 <- length(constructed_universe)
  
  constructed_universe <- union(constructed_universe, unlist(sets[[j]]))
  
  len_constructed_universe_2 <- length(constructed_universe)
  
  if (len_constructed_universe_2 > len_constructed_universe_1) {
    # the new set gave us at least one element that did not belong to the earlier cover
    cat("Set number ", j, "for set cover is : ", j, "\n")
    cat("Length of the cover is :", len_constructed_universe_2, "\n\n")
    
  }
  
  # for every attempt to add  set -- we store it
  normalized_set_cover_size_df[ctr,1] <- ctr
  normalized_set_cover_size_df[ctr,2] <- j
  normalized_set_cover_size_df[ctr,3] <- paste(unlist(sets[[j]]), collapse = "_")
  normalized_set_cover_size_df[ctr,4] <- len_constructed_universe_2
  ctr <- ctr + 1
  
  if (length(constructed_universe) == length(U)) {
    # we have all elements, so stop
    break
  }
  
}

# remove all 0 rows from the cover size

normalized_set_cover_size_df <- normalized_set_cover_size_df[normalized_set_cover_size_df$ctr != 0 & normalized_set_cover_size_df$size != 0, ]


### unnormalized tf-idf up now ####
ordered_tf_idf_indexes <- order(tf_idf, decreasing = T)

tf_idf_2 <- tf_idf[ordered_tf_idf_indexes]

set_seq_df <- data.frame(set_index = c(1:num_sets),
                         set_seq = ordered_tf_idf_indexes,
                         tf_idf_2 = tf_idf_2
)

# final ordering and stop when all elements covered

constructed_universe <- c()

set_cover_size_df <- data.frame(
  ctr = numeric(10*length(U)),
  attempted_set = numeric(10*length(U)),
  actual_set = character(10*length(U)),
  size = numeric(10*length(U)),
  stringsAsFactors = F
)

ctr <- 1

for (j in set_seq_df$set_seq) {
  
  cat("Choice number ", ctr, "\n")
  len_constructed_universe_1 <- length(constructed_universe)
  
  constructed_universe <- union(constructed_universe, unlist(sets[[j]]))
  
  len_constructed_universe_2 <- length(constructed_universe)
  
  if (len_constructed_universe_2 > len_constructed_universe_1) {
    # the new set gave us at least one element that did not belong to the earlier cover
    cat("Set number ", j, "for set cover is : ", j, "\n")
    cat("Length of the cover is :", len_constructed_universe_2, "\n\n")
    
  }
  
  # for every attempt to add  set -- we store it
  set_cover_size_df[ctr,1] <- ctr
  set_cover_size_df[ctr,2] <- j
  set_cover_size_df[ctr,3] <- paste(unlist(sets[[j]]), collapse = "_")
  set_cover_size_df[ctr,4] <- len_constructed_universe_2
  ctr <- ctr + 1
  
  if (length(constructed_universe) == length(U)) {
    # we have all elements, so stop
    break
  }
  
}

# remove all 0 rows from the cover size

set_cover_size_df <- set_cover_size_df[set_cover_size_df$ctr != 0 & set_cover_size_df$size != 0, ]

# How does this look?

ggplot(data = set_cover_size_df, aes(x = ctr, y = size)) + 
  geom_point(colour = "red", size = 2) + geom_line(colour = "black")

# plot both normalized and unnormalized tf-idf plots now

ggplot() + 
  geom_point(data = set_cover_size_df, aes(x = ctr, y = size), colour = "red") + 
  geom_line(data = set_cover_size_df, aes(x = ctr, y = size), colour = "black") + 
  geom_point(data = normalized_set_cover_size_df, aes(x = ctr, y = size), colour = "blue") + 
  geom_line(data = normalized_set_cover_size_df, aes(x = ctr, y = size), colour = "black")


# intriguing - lets get the slope of the initial straight line between log(size)
# and log(ctr)

set_cover_size_df$log_ctr <- log(set_cover_size_df$ctr)
set_cover_size_df$log_size <- log(set_cover_size_df$size)

normalized_set_cover_size_df$log_ctr <- log(normalized_set_cover_size_df$ctr)
normalized_set_cover_size_df$log_size <- log(normalized_set_cover_size_df$size)

fit_line <- lm(log_size ~ log_ctr, data = set_cover_size_df)
fit_line_normalized <- lm(log_size ~ log_ctr, data = normalized_set_cover_size_df)

restricted_fit_line <- lm(log_size ~ log_ctr, data = set_cover_size_df[set_cover_size_df$log_ctr <= 5, ])
restricted_fit_line_normalized <- lm(log_size ~ log_ctr, data = normalized_set_cover_size_df[normalized_set_cover_size_df$log_ctr <= 5, ])

# log-log plot of both

ggplot() + 
  geom_point(data = set_cover_size_df, aes(x = log_ctr, y = log_size), colour = "red") + 
  geom_line(data = set_cover_size_df, aes(x = log_ctr, y = log_size), colour = "black") + 
  geom_point(data = normalized_set_cover_size_df, aes(x = log_ctr, y = log_size), colour = "blue") + 
  geom_line(data = normalized_set_cover_size_df, aes(x = log_ctr, y = log_size), colour = "black")

# with fit lines -- so using a simple plot

ggplotRegression(restricted_fit_line)
ggplotRegression(restricted_fit_line_normalized)

# plot the tf-idf scores histogram

ggplot(data = set_seq_df, aes(x = tf_idf_2)) + 
  geom_histogram(binwidth=5, colour = "red", fill = "red")


# is the union of all the cover sets equal to the universe?
UU <- c()
for (j in 1:length(sets)) {
  v <- unlist(sets[[j]])
  
  UU <- union(UU, v)
}

