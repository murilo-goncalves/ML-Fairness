import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import distributions
import statsmodels.api as sm
from statsmodels.formula.api import logit
from fairml import audit_model
from fairml import plot_generic_dependence_dictionary
from sklearn.linear_model import LogisticRegression
pd.options.mode.chained_assignment = None

def main():
    compas_scores_two_years = pd.read_csv(r"C:\Users\marin\Desktop\UNICAMP\IC\ML-Fairness\fairness\compas_dataset\compas-scores-two-years.csv")

    compas_scores_two_years = compas_scores_two_years[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 
             'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

    compas_scores_two_years = compas_scores_two_years[((compas_scores_two_years['days_b_screening_arrest'] <=30) & 
      (compas_scores_two_years['days_b_screening_arrest'] >= -30) &
      (compas_scores_two_years['is_recid'] != -1) &
      (compas_scores_two_years['c_charge_degree'] != 'O') & 
      (compas_scores_two_years['score_text'] != 'N/A'))]

    3#X-RAY
    quant_race = compas_scores_two_years['race'].value_counts()
    quant_age = compas_scores_two_years['age'].value_counts()
    quant_score = compas_scores_two_years['score_text'].value_counts()
    quant_sex = compas_scores_two_years['sex'].value_counts()
    quant_2yr = compas_scores_two_years['two_year_recid'].value_counts()
    # print(quant_)

    tab_scoretext_race = pd.crosstab(compas_scores_two_years['score_text'],compas_scores_two_years['race'])
    tab_decilescore_race = pd.crosstab(compas_scores_two_years['decile_score'],compas_scores_two_years['race'])
    # print(tab_decilescore_race)

    distribution = sns.countplot(x='decile_score', hue='race', data=compas_scores_two_years.loc[(compas_scores_two_years['race'] == 'African-American') | (compas_scores_two_years['race'] == 'Caucasian'),:])
    # plt.show()
    plt.title("Distribution of Decile Scores by Race")
    plt.xlabel('Decile Score')
    plt.ylabel('Count')

    3#LOGISTIC REGRESSION
    categ_cols = ['score_text','age_cat','sex','race','c_charge_degree']
    compas_scores_two_years.loc[:,categ_cols] = compas_scores_two_years.loc[:,categ_cols].astype('category')

    df_Dummies = pd.get_dummies(data = compas_scores_two_years, columns=categ_cols)

    # Clean column names
    new_column_names = [col.lstrip().rstrip().lower().replace(" ", "_").replace("-", "_") for col in df_Dummies.columns]
    df_Dummies.columns = new_column_names

    # We want another variable that combines Medium and High
    df_Dummies['score_text_medhi'] = df_Dummies['score_text_medium'] + df_Dummies['score_text_high']
    
    # R-style specification
    formula = 'score_text_medhi ~ sex_female + age_cat_greater_than_45 + age_cat_less_than_25 + race_african_american + race_asian + race_hispanic + race_native_american + race_other + priors_count + c_charge_degree_m + two_year_recid'

    score_mod = logit(formula, data = df_Dummies).fit()
    # print(score_mod.summary())

    control = np.exp(-1.5255) / (1 + np.exp(-1.5255))
    #Black defendants
    black = np.exp(0.4772) / (1 - control + (control * np.exp(0.4772)))
    #Female defendants
    female = np.exp(0.2213) / (1 - control + (control * np.exp(0.2213)))
    #Younger than 25 
    younger = np.exp(1.3084) / (1 - control + (control * np.exp(1.3084)))

    ##FAIRML
    propublica_data = pd.read_csv(r"C:\Users\marin\Desktop\UNICAMP\IC\ML-Fairness\fairness\compas_dataset\propublicaCompassRecividism_data_fairml.csv\propublica_data_for_fairml.csv")

    # create feature and design matrix for model building.
    compas_rating = propublica_data.score_factor.values
    propublica_data = propublica_data.drop("score_factor", 1)

    # Train simple model
    clf = LogisticRegression(penalty='l2', C=0.01)
    clf.fit(propublica_data.values, compas_rating)

    #  call audit model with model
    total, _ = audit_model(clf.predict, propublica_data)

    # print feature importance
    print(total)

    # generate feature dependence plot
    fig = plot_dependencies(
        total.get_compress_dictionary_into_key_median(),
        reverse_values=False,
        title="FairML feature dependence"
    )
    plt.savefig("fairml_ldp.eps", transparent=False, bbox_inches='tight')
    
if (__name__ == "__main__"):
    main()
























