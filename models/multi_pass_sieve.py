import pandas as pd

class MultiPassSieveModel():
	def __init__(self, *models):
		self.models = models

	def predict(self, df):
		preds = pd.DataFrame([[False]*2]*len(df), columns=['a_coref', 'b_coref'])
		for model in self.models:
			preds_ = model.predict(df)
			preds_ = pd.DataFrame(preds_, columns=['a_coref', 'b_coref'])
			mask = ~preds['a_coref'] & ~preds['b_coref']
			preds[mask] = preds_[mask]

		return preds.values.tolist()
