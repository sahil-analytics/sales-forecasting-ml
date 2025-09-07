"""
Sales Forecasting with Predictive Analytics
A comprehensive machine learning project for predicting sales trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SalesForecaster:
    """Main class for sales forecasting project"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.models = {}
        self.predictions = {}
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic sales data for demonstration"""
        np.random.seed(42)
        
        # Date range
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]
        
        # Create features
        data = {
            'date': dates,
            'day_of_week': [d.weekday() for d in dates],
            'month': [d.month for d in dates],
            'quarter': [(d.month-1)//3 + 1 for d in dates],
            'year': [d.year for d in dates],
            'is_weekend': [1 if d.weekday() >= 5 else 0 for d in dates],
            'temperature': np.random.normal(20, 10, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'advertising_spend': np.random.uniform(100, 5000, n_samples),
            'competitor_price': np.random.uniform(50, 200, n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], n_samples),
            'promotion_type': np.random.choice(['None', 'Discount', 'BOGO', 'Seasonal'], n_samples),
            'store_location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
        }
        
        # Create target variable (sales) with logical relationships
        base_sales = 1000
        seasonal_effect = np.sin(np.arange(n_samples) * 2 * np.pi / 365) * 200
        trend = np.arange(n_samples) * 0.5
        advertising_effect = data['advertising_spend'] * 0.1
        weekend_effect = np.array([200 if w else 0 for w in data['is_weekend']])
        temperature_effect = (data['temperature'] - 20) * 5
        noise = np.random.normal(0, 50, n_samples)
        
        data['sales'] = (base_sales + seasonal_effect + trend + 
                         advertising_effect + weekend_effect + 
                         temperature_effect + noise)
        data['sales'] = np.maximum(data['sales'], 0)  # Ensure non-negative sales
        
        self.df = pd.DataFrame(data)
        print(f"✓ Generated {n_samples} samples of synthetic sales data")
        return self.df
    
    def load_data(self, file_path=None):
        """Load data from CSV file"""
        if file_path:
            self.df = pd.read_csv(file_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"✓ Loaded data from {file_path}")
        else:
            print("No file path provided. Generating sample data...")
            self.generate_sample_data()
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic info
        print("\nDataset Shape:", self.df.shape)
        print("\nColumn Types:")
        print(self.df.dtypes)
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # Sales distribution
        axes[0, 0].hist(self.df['sales'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Sales Distribution')
        axes[0, 0].set_xlabel('Sales')
        axes[0, 0].set_ylabel('Frequency')
        
        # Sales over time
        axes[0, 1].plot(self.df['date'], self.df['sales'], alpha=0.7)
        axes[0, 1].set_title('Sales Trend Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Sales')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sales by day of week
        daily_sales = self.df.groupby('day_of_week')['sales'].mean()
        axes[0, 2].bar(range(7), daily_sales.values)
        axes[0, 2].set_title('Average Sales by Day of Week')
        axes[0, 2].set_xlabel('Day (0=Monday)')
        axes[0, 2].set_ylabel('Average Sales')
        
        # Sales vs Advertising Spend
        axes[1, 0].scatter(self.df['advertising_spend'], self.df['sales'], alpha=0.5)
        axes[1, 0].set_title('Sales vs Advertising Spend')
        axes[1, 0].set_xlabel('Advertising Spend')
        axes[1, 0].set_ylabel('Sales')
        
        # Monthly sales
        monthly_sales = self.df.groupby('month')['sales'].mean()
        axes[1, 1].plot(monthly_sales.index, monthly_sales.values, marker='o')
        axes[1, 1].set_title('Average Sales by Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Sales')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Correlation heatmap (numerical features only)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_data = self.df[numerical_cols].drop(['year'], axis=1, errors='ignore').corr()
        im = axes[1, 2].imshow(correlation_data, cmap='coolwarm', aspect='auto')
        axes[1, 2].set_title('Feature Correlation Matrix')
        axes[1, 2].set_xticks(range(len(correlation_data.columns)))
        axes[1, 2].set_yticks(range(len(correlation_data.columns)))
        axes[1, 2].set_xticklabels(correlation_data.columns, rotation=45, ha='right')
        axes[1, 2].set_yticklabels(correlation_data.columns)
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.show()
        
        return self.df
    
    def prepare_features(self):
        """Feature engineering and data preparation"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # Create additional features
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['is_month_start'] = (self.df['date'].dt.day <= 7).astype(int)
        self.df['is_month_end'] = (self.df['date'].dt.day >= 24).astype(int)
        
        # Rolling averages (if we have enough data)
        if len(self.df) > 7:
            self.df['sales_rolling_7'] = self.df['sales'].rolling(window=7, min_periods=1).mean()
            self.df['sales_rolling_30'] = self.df['sales'].rolling(window=30, min_periods=1).mean()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ['product_category', 'promotion_type', 'store_location']
        
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
        
        print(f"✓ Created {len(self.df.columns)} features")
        print("\nNew features created:")
        new_features = ['day_of_year', 'week_of_year', 'is_month_start', 'is_month_end']
        if 'sales_rolling_7' in self.df.columns:
            new_features.extend(['sales_rolling_7', 'sales_rolling_30'])
        for feat in new_features:
            if feat in self.df.columns:
                print(f"  - {feat}")
        
        return self.df
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\n" + "="*60)
        print("DATA SPLITTING")
        print("="*60)
        
        # Select features for modeling
        feature_cols = ['day_of_week', 'month', 'quarter', 'is_weekend',
                       'temperature', 'humidity', 'advertising_spend', 
                       'competitor_price', 'day_of_year', 'week_of_year',
                       'is_month_start', 'is_month_end']
        
        # Add encoded categorical features
        for col in self.df.columns:
            if '_encoded' in col:
                feature_cols.append(col)
        
        # Add rolling features if they exist
        if 'sales_rolling_7' in self.df.columns:
            feature_cols.extend(['sales_rolling_7', 'sales_rolling_30'])
        
        # Filter to only existing columns
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        X = self.df[feature_cols].fillna(0)
        y = self.df['sales']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ Training set size: {len(self.X_train)} samples")
        print(f"✓ Testing set size: {len(self.X_test)} samples")
        print(f"✓ Number of features: {len(feature_cols)}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple ML models and compare performance"""
        print("\n" + "="*60)
        print("MODEL TRAINING & EVALUATION")
        print("="*60)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        results = []
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            # Store results
            self.models[name] = model
            self.predictions[name] = y_pred_test
            
            results.append({
                'Model': name,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'CV MAE': cv_mae
            })
            
            print(f"  Test MAE: {test_mae:.2f}")
            print(f"  Test RMSE: {test_rmse:.2f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  CV MAE: {cv_mae:.2f}")
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test MAE')
        
        print("\n" + "-"*60)
        print("MODEL COMPARISON SUMMARY")
        print("-"*60)
        print(results_df.to_string(index=False))
        
        # Select best model
        self.best_model = self.models[results_df.iloc[0]['Model']]
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n✓ Best Model: {best_model_name}")
        
        return results_df
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning for the best model"""
        print("\n" + "="*60)
        print(f"HYPERPARAMETER TUNING - {model_name}")
        print("="*60)
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        print("Searching for best parameters...")
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV MAE: {-grid_search.best_score_:.2f}")
        
        # Evaluate on test set
        y_pred = grid_search.predict(self.X_test_scaled)
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_r2 = r2_score(self.y_test, y_pred)
        
        print(f"Test MAE with best params: {test_mae:.2f}")
        print(f"Test R² with best params: {test_r2:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return grid_search
    
    def feature_importance(self):
        """Analyze feature importance"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.X_train.columns
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
            
            # Visualize
            plt.figure(figsize=(10, 6))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Importance')
            plt.title('Feature Importance (Top 15)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None
    
    def visualize_predictions(self):
        """Visualize model predictions"""
        print("\n" + "="*60)
        print("PREDICTION VISUALIZATION")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Predictions Analysis', fontsize=16, fontweight='bold')
        
        # Get best model predictions
        best_model_name = min(self.predictions.keys(), 
                             key=lambda k: mean_absolute_error(self.y_test, self.predictions[k]))
        y_pred = self.predictions[best_model_name]
        
        # Actual vs Predicted
        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Sales')
        axes[0, 0].set_ylabel('Predicted Sales')
        axes[0, 0].set_title(f'Actual vs Predicted - {best_model_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = self.y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Sales')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        
        # Model comparison
        model_names = list(self.predictions.keys())
        mae_scores = [mean_absolute_error(self.y_test, self.predictions[m]) for m in model_names]
        axes[1, 1].bar(range(len(model_names)), mae_scores)
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✓ Visualizations created for {len(self.predictions)} models")
    
    def make_future_predictions(self, periods=30):
        """Generate future sales predictions"""
        print("\n" + "="*60)
        print(f"FUTURE PREDICTIONS (Next {periods} days)")
        print("="*60)
        
        # Generate future dates
        last_date = self.df['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        # Create features for future dates (you'd need actual values in production)
        future_data = pd.DataFrame({
            'date': future_dates,
            'day_of_week': [d.weekday() for d in future_dates],
            'month': [d.month for d in future_dates],
            'quarter': [(d.month-1)//3 + 1 for d in future_dates],
            'is_weekend': [1 if d.weekday() >= 5 else 0 for d in future_dates],
            'day_of_year': [d.timetuple().tm_yday for d in future_dates],
            'week_of_year': pd.Series(future_dates).dt.isocalendar().week.values,
            'is_month_start': [1 if d.day <= 7 else 0 for d in future_dates],
            'is_month_end': [1 if d.day >= 24 else 0 for d in future_dates],
            # For demo, use average values for other features
            'temperature': [self.df['temperature'].mean()] * periods,
            'humidity': [self.df['humidity'].mean()] * periods,
            'advertising_spend': [self.df['advertising_spend'].mean()] * periods,
            'competitor_price': [self.df['competitor_price'].mean()] * periods,
        })
        
        # Add encoded categorical features with mode values
        for col in self.df.columns:
            if '_encoded' in col:
                future_data[col] = self.df[col].mode()[0]
        
        # Add rolling averages (use last known values)
        if 'sales_rolling_7' in self.df.columns:
            future_data['sales_rolling_7'] = self.df['sales_rolling_7'].iloc[-1]
            future_data['sales_rolling_30'] = self.df['sales_rolling_30'].iloc[-1]
        
        # Select same features as training
        feature_cols = self.X_train.columns
        X_future = future_data[feature_cols].fillna(0)
        
        # Scale features
        X_future_scaled = self.scaler.transform(X_future)
        
        # Make predictions
        future_predictions = self.best_model.predict(X_future_scaled)
        
        # Create results dataframe
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Sales': future_predictions,
            'Day_Type': ['Weekend' if d.weekday() >= 5 else 'Weekday' for d in future_dates]
        })
        
        print("\nNext 10 Days Forecast:")
        print(forecast_df.head(10).to_string(index=False))
        
        # Visualize forecast
        plt.figure(figsize=(12, 6))
        
        # Plot historical data (last 60 days)
        historical_days = 60
        hist_data = self.df.tail(historical_days)
        plt.plot(hist_data['date'], hist_data['sales'], 
                label='Historical Sales', color='blue', alpha=0.7)
        
        # Plot predictions
        plt.plot(forecast_df['Date'], forecast_df['Predicted_Sales'], 
                label='Forecasted Sales', color='red', linestyle='--', marker='o')
        
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Sales Forecast')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\n✓ Generated {periods}-day sales forecast")
        print(f"✓ Average predicted sales: ${forecast_df['Predicted_Sales'].mean():.2f}")
        print(f"✓ Total predicted sales: ${forecast_df['Predicted_Sales'].sum():.2f}")
        
        return forecast_df
    
    def save_model(self, filename='best_sales_model.pkl'):
        """Save the best model and scaler"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        # Save model
        joblib.dump(self.best_model, filename)
        print(f"✓ Model saved as '{filename}'")
        
        # Save scaler
        scaler_filename = filename.replace('.pkl', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_filename)
        print(f"✓ Scaler saved as '{scaler_filename}'")
        
        # Save feature names
        feature_filename = filename.replace('.pkl', '_features.pkl')
        joblib.dump(list(self.X_train.columns), feature_filename)
        print(f"✓ Features saved as '{feature_filename}'")
        
        return filename
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*60)
        print("GENERATING BUSINESS INSIGHTS REPORT")
        print("="*60)
        
        # Calculate key metrics
        best_model_name = min(self.predictions.keys(), 
                             key=lambda k: mean_absolute_error(self.y_test, self.predictions[k]))
        y_pred = self.predictions[best_model_name]
        
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        print("\n📊 KEY PERFORMANCE INDICATORS")
        print("-" * 40)
        print(f"Best Model: {best_model_name}")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"Root Mean Squared Error: ${rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        print(f"Forecast Accuracy: {(100 - mape):.2f}%")
        
        print("\n💡 BUSINESS INSIGHTS")
        print("-" * 40)
        
        # Weekly patterns
        weekly_avg = self.df.groupby('day_of_week')['sales'].mean()
        best_day = weekly_avg.idxmax()
        worst_day = weekly_avg.idxmin()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print(f"• Best sales day: {days[best_day]} (Avg: ${weekly_avg[best_day]:.2f})")
        print(f"• Worst sales day: {days[worst_day]} (Avg: ${weekly_avg[worst_day]:.2f})")
        
        # Monthly patterns
        monthly_avg = self.df.groupby('month')['sales'].mean()
        best_month = monthly_avg.idxmax()
        worst_month = monthly_avg.idxmin()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f"• Peak sales month: {months[best_month-1]} (Avg: ${monthly_avg[best_month]:.2f})")
        print(f"• Lowest sales month: {months[worst_month-1]} (Avg: ${monthly_avg[worst_month]:.2f})")
        
        # Advertising ROI
        if 'advertising_spend' in self.df.columns:
            correlation = self.df['advertising_spend'].corr(self.df['sales'])
            print(f"• Advertising-Sales Correlation: {correlation:.3f}")
            if correlation > 0.5:
                print("  → Strong positive correlation - advertising is effective")
            elif correlation > 0.3:
                print("  → Moderate positive correlation - advertising has some impact")
            else:
                print("  → Weak correlation - consider reviewing advertising strategy")
        
        print("\n📈 RECOMMENDATIONS")
        print("-" * 40)
        print("• Focus marketing efforts on high-performing days/months")
        print("• Optimize inventory for predicted demand patterns")
        print("• Adjust staffing based on forecasted busy periods")
        print("• Monitor and adjust for seasonal trends")
        print("• Use predictions for budget planning and resource allocation")
        
        return {
            'best_model': best_model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'accuracy': 100 - mape
        }


def main():
    """Main execution function"""
    print("="*60)
    print("SALES FORECASTING WITH PREDICTIVE ANALYTICS")
    print("="*60)
    print("Starting analysis pipeline...\n")
    
    # Initialize forecaster
    forecaster = SalesForecaster()
    
    # Step 1: Load or generate data
    print("\n[Step 1/9] Loading Data...")
    forecaster.load_data()  # Will generate sample data
    
    # Step 2: Explore data
    print("\n[Step 2/9] Exploring Data...")
    forecaster.explore_data()
    
    # Step 3: Feature engineering
    print("\n[Step 3/9] Engineering Features...")
    forecaster.prepare_features()
    
    # Step 4: Split data
    print("\n[Step 4/9] Splitting Data...")
    forecaster.split_data()
    
    # Step 5: Train models
    print("\n[Step 5/9] Training Models...")
    results = forecaster.train_models()
    
    # Step 6: Hyperparameter tuning
    print("\n[Step 6/9] Tuning Hyperparameters...")
    forecaster.hyperparameter_tuning('Random Forest')
    
    # Step 7: Feature importance
    print("\n[Step 7/9] Analyzing Feature Importance...")
    forecaster.feature_importance()
    
    # Step 8: Visualize predictions
    print("\n[Step 8/9] Visualizing Results...")
    forecaster.visualize_predictions()
    
    # Step 9: Make future predictions
    print("\n[Step 9/9] Generating Future Predictions...")
    future_forecast = forecaster.make_future_predictions(periods=30)
    
    # Generate final report
    report = forecaster.generate_report()
    
    # Save model
    forecaster.save_model()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\nProject successfully completed. Model is ready for deployment.")
    print("Check the generated visualizations and saved model files.")
    
    return forecaster, report


if __name__ == "__main__":
    # Run the complete pipeline
    forecaster, report = main()