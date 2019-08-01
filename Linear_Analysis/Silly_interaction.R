#' ---
#' title: "Silly interaction example"
#' subtitle: "Using data about condiments, and temperature and pressure"
#' author: "James Twose"
#' date: ""
#' ---

library(sjPlot)
library(sjmisc)
library(ggplot2)
library(interactions)

#+ echo=FALSE
setwd("/Users/james/Data_Science/")
#+ echo=TRUE

## Example 1 - categorical
# Read in categorical data
int_cat_df <- read.csv("Interactions_Categorical.csv")

# Show summary and head of the data set
summary(int_cat_df); str(int_cat_df); head(int_cat_df)

# run a linear regression with the interaction and enjoyment outcome
int_cat_lm <- lm(Enjoyment ~ Food*Condiment, data=int_cat_df)

# show a summary of the lm output (coefficients, t-values, p-values etc.)
summary(int_cat_lm)

# show the anova of the lm output (f-values, p-values etc.)
anova(int_cat_lm)
# here you can see that when looking at explained variance, the food:condiment interaction
# is considerably larger than either of the main effects of food or condiment.

# Be lazy and use the plot_model function to calc means and confidence intervals
p <- plot_model(int_cat_lm, type = "pred", terms = c("Food", "Condiment"))
p$data$x <- factor(p$data$x, levels = c(1,2), labels = c("Hot Dog", "Ice cream"))

# plot the interaction
ggplot(data=p$data, aes(x=x, y=predicted, group=group, colour=group)) + 
  geom_line() + 
  geom_point() +
  geom_errorbar(aes(ymin=conf.low, ymax=conf.high), width=.05) +
  labs(x = "Hot Dog vs Ice cream", y = "Food Enjoyment", colour = "Condiment") +
  ggtitle("Interaction plot for enjoyment")

# plot the main effects to compare
p <- plot_model(int_cat_lm, type = "slope", show.loess=F)
p + labs(x = "Left plot 1 = Chocolate Sauce, 2 = Mustard; Right plot 1 = Hot Dog, 2 = Ice cream", y = "Food Enjoyment", colour = "Condiment") 

# Based on the interaction plot and the main effects plot you can see that you would have missed
# a large amount of explained variance if you had not included the interaction


## Example 2 - continuous
# Read in continuous data
int_con_df <- read.csv("Interactions_Continuous.csv")

# Show summary and head of the data set
summary(int_con_df); str(int_con_df); head(int_con_df)

# run a linear regression with the interaction and enjoyment outcome
int_con_lm <- lm(Strength ~ Time + Temperature*Pressure, data=int_con_df)

# show a summary of the lm output (coefficients, t-values, p-values etc.)
summary(int_con_lm)

# show the anova of the lm output (f-values, p-values etc.)
anova(int_con_lm)

# Be lazy and use the plot_model function to plot means and confidence intervals
plot_model(int_con_lm, type = "int", terms = c("Temperature", "Pressure"))

# plot the main effects to compare
plot_model(int_con_lm, type = "slope", terms = c("Temperature", "Pressure"), show.loess=F)
plot_model(int_con_lm, type = "slope", terms = c("Temperature", "Pressure"), show.loess=TRUE)
# Although the loess lines look a bit interesting...

# Let's plot the raw figures against this "Linear trend"
p <- plot_model(int_con_lm, type = "slope", terms = c("Temperature", "Pressure"), show.loess=F)
p + geom_point()
# hmm maybe we should have used a non linear approach on this data?
