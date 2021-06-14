
import streamlit as st
import numpy as np

from sklearn.datasets import  make_moons
from sklearn.model_selection import  train_test_split
import data_helper
import matplotlib.pyplot as plt
from sthelper import StHelper
import webbrowser

##st.title('Gradient Boost Classification')
#url='https://xgboost.readthedocs.io/en/latest/parameter.html'



# import all datasets
concentric,linear,outlier,spiral,ushape,xor = data_helper.load_dataset()



# configure matplotlib styling
plt.style.use('seaborn-bright')

# Dataset selection dropdown
st.sidebar.markdown("# XgBoost Classifier")

if st.sidebar.button('HOW TO RUN'):
    st.sidebar.text('1.All the parameters have been set to \n their default values,to increase/decrease\n/input the values use the interactive widgets given below. \n'
                     '2.Once done, click on Run Algorithm\nbutton.\n'
                     '3.If the accuracy is more that 0.90\nthen you will get a celebratory balloon\n show.\n'
                     '4.To again set the default values\n, reload the page.\n'
                     )

#if st.sidebar.button('DOCUMENTATION'):
#   webbrowser.open_new_tab(url)

# dataset
dataset_options=st.sidebar.radio('Choose Dataset',('use generated dataset','use toy dataset'))

if dataset_options=='use generated dataset':
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


    def draw_meshgrid():
        a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
        b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

        XX, YY = np.meshgrid(a, b)

        input_array = np.array([XX.ravel(), YY.ravel()]).T

        return XX, YY, input_array


    # Load initial graph
    fig, ax = plt.subplots()

    # Plot initial graph
    ax.scatter(X.T[0], X.T[1], c=y, cmap='Spectral')

    orig = st.pyplot(fig)

elif dataset_options=='use toy dataset':
    dataset = st.sidebar.selectbox(
        "Dataset",
        ("U-Shaped", "Linearly Separable", "Outlier", "Two Spirals", "Concentric Circles", "XOR"), index=3
    )
    st.header(dataset)
    fig, ax = plt.subplots()
    # Plot initial graph
    df = data_helper.load_initial_graph(dataset, ax)
    orig = st.pyplot(fig)

    # Extract X and Y
    X = df.iloc[:, :2].values
    y = df.iloc[:, -1].values




#n_estimators
n_estimators = int(st.sidebar.number_input('n_estimators',min_value=0,value=100,step=10))

#use_label_encoder
#use_label_encoder=st.sidebar.radio('use_label_encoder',(True,False),index=1)

#max_depth
max_depth =int(st.sidebar.number_input('max_depth ',min_value=1,step=1,value=6))

#eta
eta=st.sidebar.slider('eta',min_value=0.0,max_value=1.0,step=0.1,value=0.3)
st.sidebar.text('Selected: {}'.format(eta))

#verbosity
verbosity = int(st.sidebar.number_input('verbosity',min_value=0,max_value=3,value=1,step=1))

#objective
objective=st.sidebar.selectbox('objective',('reg:squarederror','reg:squaredlogerror','reg:logistic','reg:pseudohubererror','binary:logistic','binary:logitraw','binary:hinge','count:poisson','survival:cox','survival:aft','aft_loss_distribution','multi:softmax','multi:softprob','rank:pairwise','rank:ndcg','rank:map','reg:gamma','reg:tweedie'),index=4)

#booster
booster=st.sidebar.radio('booster',('gbtree','gblinear','dart'),index=0)

#tree_method
tree_method=st.sidebar.selectbox('tree_method',('auto', 'exact', 'approx', 'hist', 'gpu_hist'),index=0)

#n_jobs
n_jobs=int(st.sidebar.number_input('n_jobs',min_value=0,step=1,value=1))

#gamma
gamma=float(st.sidebar.number_input('gamma',min_value=0.0,value=0.0,step=0.1))

#min_child_weight
min_child_weight=st.sidebar.number_input('min_child_weight',min_value=0.0,value=1.0)

#max_delta_step
#max_delta_step_option=st.sidebar.number_input('max_delta_step',min_value=0,value=0)
if(objective=='count:poisson'):
    max_delta_step=st.sidebar.number_input('max_delta_step',value=0.7)
else:
    max_delta_step = st.sidebar.number_input('max_delta_step', min_value=0.0, value=0.0)

#subsample
subsample=st.sidebar.slider('subsample',min_value=0.0,max_value=1.0,step=0.1,value=1.0)
st.sidebar.text('Selected: {}'.format(subsample))

#colsample_bytree
colsample_bytree=st.sidebar.slider('colsample_bytree',min_value=0.0,max_value=1.0,step=0.1,value=1.0)
st.sidebar.text('Selected: {}'.format(colsample_bytree))

#colsample_bylevel
colsample_bylevel=st.sidebar.slider('colsample_bylevel',min_value=0.0,max_value=1.0,step=0.1,value=1.0)
st.sidebar.text('Selected: {}'.format(colsample_bylevel))

#colsample_bynode
colsample_bynode=st.sidebar.slider('colsample_bynode',min_value=0.0,max_value=1.0,step=0.1,value=1.0)
st.sidebar.text('Selected: {}'.format(colsample_bynode))

#reg_alpha
reg_alpha=st.sidebar.number_input('reg_alpha',min_value=0.0,value=0.0,step=0.1)

#reg_lambda
reg_lambda=st.sidebar.number_input('reg_lambda',min_value=0.0,value=1.0,step=0.1)





#scale_pos_weight
scale_pos_weight=st.sidebar.number_input('scale_pos_weight',step=0.1,value=1.0)

#base_score
base_score=st.sidebar.number_input('base_score',value=0.5)

#random_state
random_state = int(st.sidebar.number_input('random_state', min_value=0, value=0, step=1))

#missing
missing_option=st.sidebar.radio('missing',('np.nan','set a value in float'))
if missing_option=='set a value in float':
    missing=st.sidebar.number_input('input the value')
else:
    missing=np.nan

#num_parallel_tree
num_parallel_tree=st.sidebar.number_input('num_parallel_tree',value=1,min_value=0)
#monotone_constraints

#interaction_constraints
#importance_type
importance_type=st.sidebar.selectbox('importance_type',('gain','weight','cover','total_gain','total_cover'),index=0)
#gpu_id
#validate_parameters
validate_parameters=st.sidebar.radio('validate_parameters',(True,False),index=1)








# Create sthelper object
sthelper = StHelper(X,y)

# On button click
if st.sidebar.button("RUN ALGORITHM"):



    xgboost_clf,accuracy= sthelper.train_xgboost_classifier(n_estimators,max_depth,eta,verbosity,objective,booster,tree_method,n_jobs,gamma,min_child_weight,max_delta_step,subsample,colsample_bytree,colsample_bylevel,colsample_bynode,reg_alpha,reg_lambda,scale_pos_weight,base_score,random_state ,missing,num_parallel_tree,importance_type,validate_parameters)

    sthelper.draw_main_graph(xgboost_clf,ax)
    orig.pyplot(fig)







    # plot accuracies


    st.sidebar.header("Classification Metrics")
    st.sidebar.text("XgBoost Classifier accuracy:" + str(accuracy))


    if accuracy>0.90:
       st.balloons()







