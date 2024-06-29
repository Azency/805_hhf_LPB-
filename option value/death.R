#  smo_ave_trunk：demography类型数据，平滑后的，2000-2020年，年龄从0-100+
#  AUStMoMo_female：StMoMo类型数据，平滑后的，2000-2020年，年龄从0-100+
#  AUStMoMo_male：StMoMo类型数据，平滑后的，2000-2020年，年龄从0-100+
#  plot(extract.years(forecast.bms, years = 2021:2040))

####################建立LCA模型和BMS模型
library(demography)

AU.LC <- lca(smo_ave_trunk, adjust="dt")
plot(AU.LC)
par(mfrow=c(1,2))
plot(smo_ave_trunk,years=2000:2020,ylim=c(-11,1))
plot(extract.years(forecast.lca, years = 2021:2040), ylim=c(-11,1))

AU.bms <- bms(smo_ave_trunk, breakmethod="bms")
forecast.bms <- forecast(AU.bms)
par(mfrow=c(1,2))#画图1行2副
plot(forecast.bms$kt)
plot(forecast.bms)


library(demography)
par(mfrow=c(1,3))
plot(extract.years(forecast.lca, years = 2021:2040))
plot(extract.years(forecast.bms, years = 2021:2040))
plot(smo_ave_trunk)


###################建立CBD模型,cbd(link = c("logit", "log"))
#############Binomial model with predictor: logit q[x,t] = k1[t] + f2[x] k2[t]
library(StMoMo)
# 将demography类型数据转换成“StMoMoData"
AUStMoMo_female <- StMoMoData(smo_ave_trunk, series = "female")

CBD <- cbd()
AU_female_CBD <- fit(CBD, data = central2initial(AUStMoMo_female), ages.fit = 0:100)
AU_male_CBD <- fit(CBD, data = central2initial(AUStMoMo_male), ages.fit = 0:100)
par(mfrow=c(1,2))#画图1行2副
plot(AU_male_CBD, parametricbx = FALSE)
plot(AU_female_CBD, parametricbx = FALSE)

next_year_forecast_female.cbd <- forecast(AU_female_CBD, h = 1)
> predicted_rates <- next_year_forecast_female.cbd$rates
> print(predicted_rates)

#################建立APC模型， Poisson model with predictor: log m[x,t] = a[x] + k1[t] + g[t-x]> 
APC <- apc()
wxt <- genWeightMat(AUStMoMo_female$ages,  AUStMoMo_female$years, clip = 3)
AU_female_APC <- fit(APC, data = AUStMoMo_female, wxt = wxt)
plot(AU_female_APC, parametricbx = FALSE, nCol = 3)

# 预测未来20年的死亡率
forecast.APC_female <- forecast(AU_female_APC, years = 2021:2040)
plot(forecast.APC_female, parametricbx = FALSE, nCol = 3)

###################建立模型RH， Poisson model with predictor: log m[x,t] = a[x] + b1[x] k1[t] + g[t-x]


##################################################################################################
library(demography)
library(StMoMo)

AUdata <- hmd.mx(country = "AUS", username = "Huifang.huang@mq.edu.au", password = "20221128Hu@ng", label = "Australia")
AUdata.100 = extract.ages(AUdata, ages = 0:100)
AUdata.100.trunk = extract.years(AUdata.100, years = 2000:2020)
AUdata.100.trunk.smo <- smooth.demogdata(AUdata.100.trunk)

# 将demography类型数据转换成“StMoMoData"
AUStMoMo_female.smo <- StMoMoData(AUdata.100.trunk.smo, series = "female")
AUStMoMo_male.smo <- StMoMoData(AUdata.100.trunk.smo, series = "male")


##建立LC模型
LC <- lca(AUdata.100.trunk.smo, adjust="dt")
plot(LC)
par(mfrow=c(1,2))
plot(AUdata.100.trunk.smo,years=2000:2020,ylim=c(-11,1))
forecast.lc <- forecast(LC, jumpchoice="actual")
plot(extract.years(forecast.lc, years = 2021:2040), ylim=c(-11,1))

######LC模型使用bms方法计算
bms <- bms(AUdata.100.trunk.smo, breakmethod="bms")
forecast.bms <- forecast(bms)
par(mfrow=c(1,2))#画图1行2副
plot(forecast.bms$kt)
plot(extract.years(forecast.bms, years = 2021:2040), ylim=c(-11,1))


par(mfrow=c(1,3))
plot(extract.years(forecast.lc, years = 2021:2040))
plot(extract.years(forecast.bms, years = 2021:2040))
plot(AUdata.100.trunk.smo)



#############CBD模型，Binomial model with predictor: logit q[x,t] = k1[t] + f2[x] k2[t]
CBD <- cbd()
CBD.female <- fit(CBD, data = central2initial(AUStMoMo_female.smo), ages.fit = 50:100)
CBD.male <- fit(CBD, data = central2initial(AUStMoMo_male.smo), ages.fit = 50:100)
par(mfrow=c(1,2))#画图1行2副
plot(CBD.female, parametricbx = FALSE)
plot(CBD.male, parametricbx = FALSE)

forecast.cbd.female <- forecast(CBD.female, h = 20)
forecast.cbd.male <- forecast(CBD.male, h = 20)

forecast.cbd.female.rates <- forecast.cbd.female$rates
forecast.cbd.male.rates <- forecast.cbd.male$rates


#################建立APC模型， Poisson model with predictor: log m[x,t] = a[x] + k1[t] + g[t-x]> 
APC <- apc()
wxt <- genWeightMat(AUStMoMo_female.smo$ages,  AUStMoMo_female.smo$years, clip = 3)
APC.female <- fit(APC, data = AUStMoMo_female.smo, wxt = wxt)
plot(APC.female, parametricbx = FALSE, nCol = 3)

forecast.APC.female <- forecast(APC.female, years = 2021:2040)# 预测未来20年的死亡率
plot(forecast.APC.female, parametricbx = FALSE, nCol = 3)
