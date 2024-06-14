
library(lmerTest)
library(ordinal)
library(emmeans)
library(carData)
library(car)
library(readxl)
library(glue)


exptype <- "exp2"
dep_var <- "rt" # ifcorr rt
data <- read.csv(glue("C:\\Users\\18357\\Desktop\\GradThesis\\csv\\exp_all.csv"))

data <- subset(data, data$exp_type == exptype)

data <- subset(data, data$ifanimal == "TRUE")
# data <- subset(data, data$word == "elephant" | data$word == "kangaroo" | data$word == "turtle")
if (dep_var == "rt") {
    data <- subset(data, data$ifcorr == 1)
}
data$rt <- data$rt * 1000

#mean(subset(data, data$priming == "primingeq" & data$syl == 3)$rt)
#sd(subset(data, data$priming == "primingeq" & data$syl == 3)$rt)
#data$rt <- log(data$rt, 2)

data$Tpriming <- ifelse(data$priming == "priming", -0.5, 0.5)
data$Tsyl <- ifelse(data$syl == 2, -0.5, 0.5)
data$Texp_type <- ifelse(data$exp_type == "exp1", -0.5, 0.5)

data$Tpriming <- factor(data$Tpriming)
data$Texp_type <- factor(data$Texp_type)
data$Tsyl <- factor(data$Tsyl)



print("Fitting Started √")
# +Tpriming + Tsyl + Texp_type + Tpriming:Tsyl + Tpriming:Texp_type + Tsyl:Texp_type + Tpriming:Tsyl:Texp_type
# +Tpriming + Tsyl + Tpriming:Tsyl
if (dep_var == "rt") {

    model1 <<- lmer("rt ~ Tpriming*Tsyl + (1  |sub) + (1  |word) +(1 |familiarity)", REML=TRUE, data=data)


    print(anova(model1, type=3, ddf="Kenward-Roger"))

    summary(model1)

} else {

    #model1 <<- glmer("ifcorr ~ Tpriming*Tsyl + (1 + Tsyl + Texp_type + Tpriming:Tsyl + Tpriming:Texp_type + Tsyl:Texp_type + Tpriming:Tsyl:Texp_type|sub) + (1+Tpriming + Tsyl + Texp_type + Tpriming:Tsyl + Tpriming:Texp_type + Tsyl:Texp_type + Tpriming:Tsyl:Texp_type |word) +(1|familiarity)", family=binomial, data=data)
    model1 <<- glmer("ifcorr ~ Tpriming*Tsyl + (1  |sub) + (1   |word) +(1    |familiarity)", family=binomial, data=data)

    print(car::Anova(model1, type=3, test.statistic="Chisq"))

    summary(model1)

}

emmeans(model1, pairwise ~ Tsyl)

#emmeans(model1, pairwise ~ Texp_type|Tsyl)

#emmeans(model1, pairwise ~ Tsyl|Texp_type)

# emmeans::contrast(emmeans(model1, specs="Texp_type", by="Tsyl"),"pairwise", adjust="bonferroni")

print("sub")
sort(attr(VarCorr(model1)$sub, "stddev"), decreasing = F)[1]

print("word")
sort(attr(VarCorr(model1)$word, "stddev"), decreasing = F)[1]

print("familiarity")
sort(attr(VarCorr(model1)$familiarity, "stddev"), decreasing = F)[1]


