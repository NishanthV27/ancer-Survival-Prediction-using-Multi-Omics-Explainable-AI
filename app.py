import os
import json
import traceback
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Global model storage
_models = None
_dataset = None

def get_dataset():
    """Load and cache dataset."""
    global _dataset
    if _dataset is None:
        try:
            from ml.preprocessing import load_dataset
            _dataset = load_dataset(app.config['DATASET_PATH'])
        except Exception as e:
            return None, str(e)
    return _dataset, None

def get_models():
    """Load and cache trained models."""
    global _models
    if _models is None:
        try:
            from ml.train_models import load_models
            _models = load_models(app.config['MODELS_DIR'])
        except Exception as e:
            return None, str(e)
    return _models, None


@app.route('/')
def index():
    """Home page with project overview."""
    df, error = get_dataset()
    stats = {}
    if df is not None:
        stats = {
            'total_patients': len(df),
            'events': int(df['event'].sum()),
            'censored': int((df['event'] == 0).sum()),
            'avg_age': round(df['age'].mean(), 1),
        }
    return render_template('index.html', stats=stats, data_error=error)


@app.route('/dashboard')
def dashboard():
    """Dashboard with dataset statistics."""
    df, error = get_dataset()
    stats = {}
    if df is not None:
        stats = {
            'total_patients': len(df),
            'events': int(df['event'].sum()),
            'censored': int((df['event'] == 0).sum()),
            'avg_age': round(df['age'].mean(), 1),
            'avg_survival': round(df['survival_months'].mean(), 1),
            'median_survival': round(df['survival_months'].median(), 1),
            'tumor_stage_dist': df['tumor_stage'].value_counts().to_dict(),
            'age_stats': {
                'min': int(df['age'].min()),
                'max': int(df['age'].max()),
                'mean': round(df['age'].mean(), 1),
                'std': round(df['age'].std(), 1)
            }
        }
    return render_template('dashboard.html', stats=stats, data_error=error)


@app.route('/predict')
def predict_page():
    """Prediction form page."""
    df, _ = get_dataset()
    input_ranges = {}
    if df is not None:
        input_ranges = {
            'age': {'min': int(df['age'].min()), 'max': int(df['age'].max())},
            'gene_expr_A': {'min': round(float(df['gene_expr_A'].min()), 2), 'max': round(float(df['gene_expr_A'].max()), 2)},
            'gene_expr_B': {'min': round(float(df['gene_expr_B'].min()), 2), 'max': round(float(df['gene_expr_B'].max()), 2)},
            'gene_expr_C': {'min': round(float(df['gene_expr_C'].min()), 2), 'max': round(float(df['gene_expr_C'].max()), 2)},
            'methylation_A': {'min': 0.0, 'max': 1.0},
            'methylation_B': {'min': 0.0, 'max': 1.0},
        }
    return render_template('predict.html', input_ranges=input_ranges)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for risk prediction."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate required fields
        required = ['age', 'tumor_stage', 'gene_expr_A', 'gene_expr_B', 'gene_expr_C', 'methylation_A', 'methylation_B']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
        
        # Validate types and ranges
        try:
            patient_data = {
                'age': float(data['age']),
                'tumor_stage': str(data['tumor_stage']),
                'gene_expr_A': float(data['gene_expr_A']),
                'gene_expr_B': float(data['gene_expr_B']),
                'gene_expr_C': float(data['gene_expr_C']),
                'methylation_A': float(data['methylation_A']),
                'methylation_B': float(data['methylation_B']),
            }
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
        
        # Validate tumor stage
        if patient_data['tumor_stage'] not in ['I', 'II', 'III', 'IV']:
            return jsonify({'error': 'Tumor stage must be I, II, III, or IV'}), 400
        
        # Validate ranges
        if not (0 <= patient_data['age'] <= 120):
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        if not (0 <= patient_data['methylation_A'] <= 1):
            return jsonify({'error': 'Methylation A must be between 0 and 1'}), 400
        if not (0 <= patient_data['methylation_B'] <= 1):
            return jsonify({'error': 'Methylation B must be between 0 and 1'}), 400
        
        models, error = get_models()
        if error:
            return jsonify({'error': f'Models not loaded: {error}. Please train models first by running: python -m ml.train_models'}), 503
        
        from ml.prediction import predict_risk
        model_name = data.get('model', 'rsf')
        result = predict_risk(patient_data, models, model_name=model_name)
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f'Prediction error: {traceback.format_exc()}')
        return jsonify({'error': 'An internal error occurred during prediction. Please check server logs.'}), 500


@app.route('/analytics')
def analytics_page():
    """Analytics and Kaplan-Meier page."""
    return render_template('analytics.html')


@app.route('/api/analytics', methods=['GET'])
def api_analytics():
    """API endpoint for analytics data."""
    try:
        df, error = get_dataset()
        if error:
            return jsonify({'error': error}), 500
        
        models, model_error = get_models()
        
        from ml.survival_analysis import get_km_plot_data, compute_km_by_feature
        
        result = {
            'km_data': None,
            'km_by_stage': None,
            'model_available': models is not None
        }
        
        if models is not None:
            try:
                km_data = get_km_plot_data(df, models)
                result['km_data'] = km_data
            except Exception as e:
                result['km_error'] = str(e)
        else:
            result['km_error'] = f'Models not trained: {model_error}'
        
        try:
            km_by_stage = compute_km_by_feature(df, 'tumor_stage')
            result['km_by_stage'] = km_by_stage
        except Exception as e:
            result['km_stage_error'] = str(e)
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f'Analytics error: {traceback.format_exc()}')
        return jsonify({'error': 'An internal error occurred.'}), 500


@app.route('/explainability')
def explainability_page():
    """Explainable AI page."""
    df, _ = get_dataset()
    input_ranges = {}
    if df is not None:
        input_ranges = {
            'age': {'min': int(df['age'].min()), 'max': int(df['age'].max())},
            'gene_expr_A': {'min': round(float(df['gene_expr_A'].min()), 2), 'max': round(float(df['gene_expr_A'].max()), 2)},
            'gene_expr_B': {'min': round(float(df['gene_expr_B'].min()), 2), 'max': round(float(df['gene_expr_B'].max()), 2)},
            'gene_expr_C': {'min': round(float(df['gene_expr_C'].min()), 2), 'max': round(float(df['gene_expr_C'].max()), 2)},
            'methylation_A': {'min': 0.0, 'max': 1.0},
            'methylation_B': {'min': 0.0, 'max': 1.0},
        }
    return render_template('explainability.html', input_ranges=input_ranges)


@app.route('/api/explainability', methods=['GET', 'POST'])
def api_explainability():
    """API endpoint for explainability data."""
    try:
        models, error = get_models()
        if error:
            return jsonify({'error': f'Models not loaded: {error}'}), 503
        
        from ml.explainability import (
            compute_shap_values, get_global_importance,
            get_local_explanation, get_feature_descriptions,
            compute_permutation_importance
        )
        from ml.preprocessing import load_dataset, preprocess_data
        
        result = {
            'feature_descriptions': get_feature_descriptions(),
        }
        
        metadata = models.get('metadata', {})
        preprocessor_info = models.get('preprocessor_info', {})
        feature_names = preprocessor_info.get('feature_names', metadata.get('feature_names', []))
        best_model_name = metadata.get('best_model', 'rsf')
        model = models.get(best_model_name, models.get('rsf'))
        
        # Global importance: compute SHAP values first, then rank
        try:
            df = load_dataset(app.config['DATASET_PATH'])
            X_train, X_test, y_train, y_test, _ = preprocess_data(df, random_seed=42, test_size=0.2)
            
            shap_result = compute_shap_values(model, X_train, X_test, model_name=best_model_name)
            global_list = get_global_importance(shap_result['shap_values'], feature_names)
            
            # Format for frontend: lists of features and importances
            result['global_importance'] = {
                'method': shap_result.get('method', 'unknown'),
                'features': [item['feature'] for item in global_list],
                'importances': [item['importance'] for item in global_list],
                'details': global_list,
            }
        except Exception as e:
            # Fallback to permutation importance
            try:
                df = load_dataset(app.config['DATASET_PATH'])
                X_train, X_test, y_train, y_test, _ = preprocess_data(df, random_seed=42, test_size=0.2)
                perm_result = compute_permutation_importance(model, X_test, y_test, feature_names)
                result['global_importance'] = {
                    'method': 'Permutation Importance (C-index drop)',
                    'features': [item['feature'] for item in perm_result['ranked_features']],
                    'importances': [item['importance'] for item in perm_result['ranked_features']],
                    'details': perm_result['ranked_features'],
                }
            except Exception as e2:
                result['global_importance_error'] = f'SHAP: {str(e)}; Permutation: {str(e2)}'
        
        # Local explanation if patient data provided
        if request.method == 'POST':
            data = request.get_json()
            if data and 'patient' in data:
                try:
                    patient_data = {
                        'age': float(data['patient']['age']),
                        'tumor_stage': str(data['patient']['tumor_stage']),
                        'gene_expr_A': float(data['patient']['gene_expr_A']),
                        'gene_expr_B': float(data['patient']['gene_expr_B']),
                        'gene_expr_C': float(data['patient']['gene_expr_C']),
                        'methylation_A': float(data['patient']['methylation_A']),
                        'methylation_B': float(data['patient']['methylation_B']),
                    }
                    model_name = data.get('model', 'rsf')
                    selected_model = models.get(model_name, model)
                    local_exp = get_local_explanation(
                        patient_data, selected_model, preprocessor_info, model_name=model_name
                    )
                    # Format for frontend
                    attribs = local_exp.get('feature_attributions', [])
                    result['local_explanation'] = {
                        'method': 'Perturbation Attribution' if model_name == 'rsf' else 'Linear Coefficient Attribution',
                        'features': [a['feature'] for a in attribs],
                        'contributions': [a['contribution'] for a in attribs],
                        'details': attribs,
                    }
                except Exception as e:
                    result['local_explanation_error'] = str(e)
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f'Explainability error: {traceback.format_exc()}')
        return jsonify({'error': 'An internal error occurred.'}), 500


@app.route('/dataset')
def dataset_page():
    """Dataset explorer page."""
    return render_template('dataset.html')


@app.route('/api/dataset', methods=['GET'])
def api_dataset():
    """API endpoint for dataset exploration."""
    try:
        df, error = get_dataset()
        if error:
            return jsonify({'error': error}), 500
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        per_page = min(per_page, 1000)  # Max 1000 per page
        
        # Filters
        search = request.args.get('search', '').strip()
        stage_filter = request.args.get('stage', '').strip()
        event_filter = request.args.get('event', '').strip()
        
        filtered = df.copy()
        
        if search:
            filtered = filtered[filtered['patient_id'].str.contains(search, case=False, na=False)]
        if stage_filter:
            clean_stage = stage_filter.replace('Stage ', '').strip()
            filtered = filtered[filtered['tumor_stage'].isin([stage_filter, f"Stage {clean_stage}", clean_stage])]
        if event_filter != '':
            try:
                filtered = filtered[filtered['event'] == int(event_filter)]
            except ValueError:
                pass
        
        total = len(filtered)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))
        
        start = (page - 1) * per_page
        end = start + per_page
        page_data = filtered.iloc[start:end]
        
        # Summary stats
        summary = {
            'total_records': len(df),
            'filtered_records': total,
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'missing_values': df.isnull().sum().to_dict(),
            'event_distribution': df['event'].value_counts().to_dict(),
            'stage_distribution': df['tumor_stage'].value_counts().to_dict(),
        }
        
        return jsonify({
            'data': page_data.to_dict('records'),
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'total_pages': total_pages,
            },
            'summary': summary,
        })
    
    except Exception as e:
        app.logger.error(f'Dataset error: {traceback.format_exc()}')
        return jsonify({'error': 'An internal error occurred.'}), 500


@app.route('/api/dataset/download')
def download_dataset():
    """Download the active dataset CSV."""
    try:
        filepath = app.config['DATASET_PATH']
        if not os.path.exists(filepath):
            return jsonify({'error': 'Dataset file not found'}), 404
        filename = os.path.basename(filepath)
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/multiomics/status')
def api_multiomics_status():
    """Report real multi-omics data integration status."""
    meta_path = os.path.join(app.config['DATA_DIR'], 'processed', 'tcga_coad_multiomics_metadata.json')
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            pass
    return jsonify({
        'status': 'ok',
        'dataset_mode': getattr(Config, 'DATASET_MODE', 'real_multiomics'),
        'dataset_path': app.config['DATASET_PATH'],
        'multiomics_metadata': meta,
        'biomarkers': [
            {'id': 'gene_expr_A', 'name': 'TP53', 'type': 'RNA-seq', 'metric': 'log2(TPM+1)'},
            {'id': 'gene_expr_B', 'name': 'KRAS', 'type': 'RNA-seq', 'metric': 'log2(TPM+1)'},
            {'id': 'gene_expr_C', 'name': 'MUC2', 'type': 'RNA-seq', 'metric': 'log2(TPM+1)'},
            {'id': 'methylation_A', 'name': 'cg00000292', 'type': 'DNA Methylation', 'metric': 'Beta-value'},
            {'id': 'methylation_B', 'name': 'cg00005847', 'type': 'DNA Methylation', 'metric': 'Beta-value'},
        ]
    })


@app.route('/about')
def about_page():
    """About project page."""
    return render_template('about.html')


@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    health = {
        'status': 'ok',
        'dataset': False,
        'models': False,
        'dataset_mode': getattr(Config, 'DATASET_MODE', 'unknown'),
        'dataset_path': app.config['DATASET_PATH'],
    }
    
    df, err = get_dataset()
    if df is not None:
        health['dataset'] = True
        health['dataset_records'] = len(df)
    else:
        health['dataset_error'] = err
    
    models, err = get_models()
    if models is not None:
        health['models'] = True
        health['model_names'] = list(models.get('trained_models', {}).keys())
    else:
        health['models_error'] = err
    
    return jsonify(health)


@app.route('/api/dashboard', methods=['GET'])
def api_dashboard():
    """API endpoint for dashboard chart data."""
    try:
        df, error = get_dataset()
        if error:
            return jsonify({'error': error}), 500
        
        # Age distribution (binned)
        age_bins = pd.cut(df['age'], bins=[20, 30, 40, 50, 60, 70, 80, 90], right=False)
        age_dist = age_bins.value_counts().sort_index()
        
        # Survival distribution
        surv_bins = pd.cut(df['survival_months'], bins=10)
        surv_dist = surv_bins.value_counts().sort_index()
        
        # Gene expression stats
        gene_cols = ['gene_expr_A', 'gene_expr_B', 'gene_expr_C']
        gene_stats = {}
        for col in gene_cols:
            gene_stats[col] = {
                'mean': round(float(df[col].mean()), 2),
                'std': round(float(df[col].std()), 2),
                'min': round(float(df[col].min()), 2),
                'max': round(float(df[col].max()), 2),
                'median': round(float(df[col].median()), 2),
            }
        
        # Methylation stats  
        meth_cols = ['methylation_A', 'methylation_B']
        meth_stats = {}
        for col in meth_cols:
            meth_stats[col] = {
                'mean': round(float(df[col].mean()), 2),
                'std': round(float(df[col].std()), 2),
                'min': round(float(df[col].min()), 2),
                'max': round(float(df[col].max()), 2),
                'median': round(float(df[col].median()), 2),
            }
        
        # Model metrics if available
        model_metrics = None
        models, _ = get_models()
        if models and 'metadata' in models:
            model_metrics = models['metadata'].get('metrics', {})
        
        return jsonify({
            'age_distribution': {
                'labels': [str(b) for b in age_dist.index],
                'values': age_dist.values.tolist(),
            },
            'tumor_stage_distribution': df['tumor_stage'].value_counts().to_dict(),
            'event_distribution': {'events': int(df['event'].sum()), 'censored': int((df['event'] == 0).sum())},
            'survival_distribution': {
                'labels': [str(b) for b in surv_dist.index],
                'values': surv_dist.values.tolist(),
            },
            'gene_expression_stats': gene_stats,
            'methylation_stats': meth_stats,
            'model_metrics': model_metrics,
            'basic_stats': {
                'total_patients': len(df),
                'avg_age': round(float(df['age'].mean()), 1),
                'avg_survival': round(float(df['survival_months'].mean()), 1),
                'median_survival': round(float(df['survival_months'].median()), 1),
            }
        })
    except Exception as e:
        app.logger.error(f'Dashboard API error: {traceback.format_exc()}')
        return jsonify({'error': 'An internal error occurred.'}), 500


if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='127.0.0.1', port=5000)
