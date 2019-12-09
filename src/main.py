def outlier_observation(X_pr, target, itters):

  X_pr.index.name = 'index'

  da =0
  leng = itters
  for r in range(leng):
    da += 1
    first = X_pr.sample(int(len(X_pr)/2))
    second = X_pr[~X_pr.isin(first)].dropna()

    d_train = lgb.Dataset(first.drop(columns=[target]), label=np.log1p(first[target]))
    d_valid = lgb.Dataset(second.drop(columns=[target]), label=np.log1p(second[target]))

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmsle',
        'max_depth': 6, 
        'learning_rate': 0.1,
        'verbose': 0,
      'num_threads':16}
    n_estimators = 100

    model = lgb.train(params, d_train, 100, verbose_eval=1)

    preds = model.predict(second)

    #predictions = np.clip(preds, df[target].min(), df[target].max())

    second[target+ "_prediction"] = np.expm1(preds) # predictions
    second["real_"+target] = second[target]
    second["over_prediction_percentage"] = (second[target+ "_prediction"]-second["real_"+target])/second["real_"+target]


    model = lgb.train(params, d_valid, 100, verbose_eval=1)

    preds = model.predict(first)

    #predictions = np.clip(preds, df[target].min(), df[target].max())

    first[target+ "_prediction"] = np.expm1(preds) # predictions
    first["real_"+target] = first[target]
    first["over_prediction_percentage"] = (first[target+ "_prediction"]-first["real_"+target])/first["real_"+target]
    


    final = pd.concat((first, second), axis=0)

    if da==1:
      framed = final.sort_index()
    else:
      framed["over_prediction_percentage"] = framed["over_prediction_percentage"].sort_index() + final["over_prediction_percentage"].sort_index()


  framed = framed.sort_values("over_prediction_percentage",ascending=False)

  framed = framed.replace([np.inf, -np.inf], np.nan).dropna(subset=["over_prediction_percentage"], how="all") # not needed but safe

  #print(second.reset_index())

  high = framed.reset_index().head()[["index","over_prediction_percentage"]]
  high.columns = ["Overprediction Index", "Overpredict Percentage"]
  high["Overpredict Percentage"] = (high["Overpredict Percentage"]*100/leng).astype(int)

  low = framed.iloc[::-1].reset_index().head()[["index","over_prediction_percentage"]]
  low.columns = ["Underprediction Index", "Underpredict Percentage"]
  low["Underpredict Percentage"] = (low["Underpredict Percentage"]*100/leng).astype(int)

  together = pd.merge(high, low,left_index=True, right_index=True,how="left")


  return together, framed, framed.index


def feature_calcs(second, target, original):

  overpred_target = "over_prediction_percentage"

  if original:
    cols_drop = [ overpred_target, target+ "_prediction", "real_"+target,target ]
    d_second = lgb.Dataset(second.drop(columns=cols_drop), label=second[target])
  else:
    cols_drop = [overpred_target,  target+ "_prediction", "real_"+target]
    d_second = lgb.Dataset(second.drop(columns=cols_drop), label=second[overpred_target])


  #cols_drop = [overpred_target, target+ "_prediction", "real_"+target]

  f = -1
  for seeds in [15,1,6,7,2]:
      f = f +1
      print("Training Iteration: "+ str(f+1)+"/"+str(5))

      params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmsle',
        'max_depth': 6, 
        'learning_rate': 0.1,
        'verbose': 0,
      'num_threads':16}
      n_estimators = 100

      params["feature_fraction_seed"] = seeds
      
      params["random_seed"] = seeds + 1
      model = lgb.train(params, d_second, verbose_eval=1000)


      shap_values = shap.TreeExplainer(model).shap_values(second.drop(cols_drop, axis=1))

      shap_fram = pd.DataFrame(shap_values[:,:], columns=list(second.drop(cols_drop, axis=1).columns))

      shap_new = shap_fram.sum().sort_values().to_frame()

      shap_new.columns = ["SHAP"]
      if original:
        shap_new["SHAP_abs"] = shap_new["SHAP"].abs()
      
      if f==0:
          shap_fin = shap_new
      else:
          shap_fin = shap_fin + shap_new

  shap_fin = shap_fin.sort_values("SHAP", ascending=False)

  return shap_fin

def feature_frame(second, target):
 
  print("First Half")
  feature_over = feature_calcs(second,target,False)

  feature_over = feature_over[~feature_over.index.isin([target])]
  feature_over = feature_over.reset_index()

  low = feature_over.reset_index(drop=True).head()
  low.columns = ["Larger Feature Leads to Underprediction (FLU)", "FLU Value"]
  low["FLU Value"] = low["FLU Value"].abs()

  print("...........")

  high = feature_over.iloc[::-1].reset_index(drop=True).head()
  high.columns = ["Larger Feature Leads to Overprediction (FLO)", "FLO Value"]
  high["FLO Value"] = high["FLO Value"].abs()

  print("Second Half")
  feature_org = feature_calcs(second,target,True)
  top = feature_org.sort_values("SHAP_abs",ascending=False).drop(columns=["SHAP"]).reset_index().head()
  top.columns = ["Top Feature", "ABS SHAP Value"]

  new = pd.merge(top, high,left_index=True, right_index=True,how="left")
  new = pd.merge(new, low,left_index=True, right_index=True,how="left")

  return new 

def outliers(X_pr):
  targets = ["number_of_reviews","price"]


  targets = list(X_pr.columns)
  ka = 0 
  for target in targets:
    ka += 1

    print("Start " + target + " ("+str(ka)+"/"+str(len(targets))+")")

    together, framed, ind = outlier_observation(X_pr, target, 5)
    try:
      frame = feature_frame(framed,target)
    except:
      print("Bad Feature")
      continue
    unit = pd.merge(together, frame,left_index=True, right_index=True,how="left")
    unit.insert(loc=4, column="Predicted Feature", value=target)

    if ka==1:
      full = unit
    else:
      full = pd.concat((full, unit),axis=0)

    print(" ")
    print("Completed " + target + " ("+str(ka)+"/"+str(len(targets))+")")
    print("=================== ")
  return full

def features(full):

  des = full.groupby("Predicted Feature")["Predicted Feature"].count().to_frame().rename(columns={'Predicted Feature':'count'}); des.head()
  des.index.names = ["Features"]

  predictability = full.groupby("Predicted Feature")["ABS SHAP Value"].mean().sort_values(ascending=False); predictability
  des = pd.merge(des,predictability, left_index=True, right_index=True, how="left" ).rename(columns={'ABS SHAP Value':'predictability'}); des.head()

  informativeness = full.groupby("Top Feature")["ABS SHAP Value"].mean().sort_values(ascending=False)
  des = pd.merge(des,informativeness, left_index=True, right_index=True, how="left" ).rename(columns={'ABS SHAP Value':'informativeness'}); des.head()

  #observation_overprediction_errors = full.groupby("Overprediction Index")["Overprediction Index"].count().sort_values(ascending=False);observation_overprediction_errors

  #observation_underprediction_errors = full.groupby("Underprediction Index")["Underprediction Index"].count().sort_values(ascending=False)

  deceptive_overpredicting = full.groupby("Larger Feature Leads to Overprediction (FLO)")["FLO Value"].mean().sort_values(ascending=False); deceptive_overpredicting
  des = pd.merge(des,deceptive_overpredicting, left_index=True, right_index=True, how="left" ).rename(columns={'FLO Value':'overpredictor'}); des.head()


  deceptive_underpredicting = full.groupby("Larger Feature Leads to Underprediction (FLU)")["FLU Value"].mean().sort_values(ascending=False); deceptive_underpredicting
  des = pd.merge(des,deceptive_underpredicting, left_index=True, right_index=True, how="left" ).rename(columns={'FLU Value':'underpredictor'}); des.head()

  prediction_volatility = full.groupby("Predicted Feature")[["Overpredict Percentage","Underpredict Percentage"]].mean(); prediction_volatility
  prediction_volatility = prediction_volatility["Overpredict Percentage"].abs() + prediction_volatility["Underpredict Percentage"].abs()
  prediction_volatility = prediction_volatility.sort_values(ascending=False); prediction_volatility.head()


  des = pd.merge(des,prediction_volatility.to_frame().rename(columns={0:'outlier_driver'}), left_index=True, right_index=True, how="left" ); des.head()

  del des["count"]
  des.fillna(value=0, inplace=True)
  add = pd.DataFrame(index=range(5))
  for col in des.columns:
    here = des.sort_values(col,ascending=False)[col].head()
    add[col + " Feature"] = here.index
    add[col + " Value"] = here.values

  return full, add

def tables(df):
  return features(outliers(df))
