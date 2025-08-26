class CoinMarketCapAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.market_data = None
        self.forecast_model = None

    def fetch_data(self):
        from data.fetch_data import get_market_data
        self.market_data = get_market_data(self.api_key)

    def train_model(self):
        from models.forecast_model import ForecastModel
        self.forecast_model = ForecastModel()
        self.forecast_model.train(self.market_data)

    def make_forecast(self):
        if self.forecast_model is None:
            raise Exception("Model has not been trained yet.")
        return self.forecast_model.predict()

    def run(self):
        self.fetch_data()
        self.train_model()
        predictions = self.make_forecast()
        return predictions