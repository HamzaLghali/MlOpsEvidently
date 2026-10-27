from sklearn import datasets
from sklearn import ensemble
from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestAccuracyScore,
    TestPrecisionByClass,
    TestRecallByClass,
    TestF1ByClass,
    TestColumnNumberOfMissingValues
)
import json


TARGET_COLUMN = "target"
PREDICTION_COLUMN = "prediction"


bcancer_data = datasets.load_breast_cancer(as_frame=True)
bcancer_df = bcancer_data.frame
feature_names = bcancer_data.feature_names.tolist()

bcancer_ref = bcancer_df.sample(n=300, random_state=42)
bcancer_cur = bcancer_df.drop(bcancer_ref.index).sample(n=200, random_state=42)

model = ensemble.RandomForestClassifier(random_state=1, n_estimators=5, max_depth=3)
model.fit(bcancer_ref[feature_names], bcancer_ref[TARGET_COLUMN])

bcancer_ref[PREDICTION_COLUMN] = model.predict_proba(bcancer_ref[feature_names])[:, 1]
bcancer_cur[PREDICTION_COLUMN] = model.predict_proba(bcancer_cur[feature_names])[:, 1]

column_mapping = ColumnMapping()
column_mapping.target = TARGET_COLUMN
column_mapping.prediction = PREDICTION_COLUMN

print("=== TESTS EVIDENTLY ===")


test_suite = TestSuite(tests=[
    
    TestAccuracyScore(gte=0.85),
    TestPrecisionByClass(label=0, gte=1),  
    TestPrecisionByClass(label=1, gte=0.80),  
    TestRecallByClass(label=0, gte=0.80),     
    TestRecallByClass(label=1, gte=0.80),     
    TestF1ByClass(label=0, gte=0.80),         
    TestF1ByClass(label=1, gte=0.80),         
    TestColumnNumberOfMissingValues(column_name=TARGET_COLUMN, eq=0),
])


test_suite.run(reference_data=bcancer_ref,
               current_data=bcancer_cur,
               column_mapping=column_mapping)


test_suite.show(mode='inline')


test_suite.save_html("reports/test_results.html")
print("Rapport HTML sauvegardé: test_results.html")


print("\n=== RÉSULTATS DES TESTS ===")
results_json = test_suite.json()
results_dict = json.loads(results_json)

if 'tests' in results_dict:
    passed_tests = 0
    failed_tests = 0
    
    for test in results_dict['tests']:
        test_name = test.get('name', 'Test sans nom')
        test_status = test.get('status', 'Statut inconnu')
        
        if test_status == 'SUCCESS':
            passed_tests += 1
            print(f" {test_name}: RÉUSSI")
        else:
            failed_tests += 1
            print(f" {test_name}: ÉCHOUÉ")
        
        if 'description' in test:
            print(f"    {test['description']}")
        print()
    
    
    print(f" RÉSUMÉ: {passed_tests} tests réussis, {failed_tests} tests échoués")
