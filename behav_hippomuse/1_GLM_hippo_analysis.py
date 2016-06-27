from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy
from scipy import stats
import statsmodels.formula.api as smfrmla
import statsmodels.api as sm
import xml.etree.ElementTree as ET
import statsmodels.sandbox.stats.multicomp as multicomp
import glob
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET
from sklearn.externals import joblib
import copy
import itertools
from nemenyi import kw_nemenyi
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from numpy import *
import matplotlib.patches as mpatches

def read_roiLabel(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    names = []

    for field in root.findall('label'):
        label = field.get('id')
        if int(label) != 0:
            name = field.get('fullname')
            # print label, name
            labels.append(int(label))
            names.append(str(name))

    return labels, names



def plot_hist(roi_betas, stat, score, component = 1, title="GLM"):

    labels, names = read_roiLabel(os.path.join('lpba40.label.xml'))
    names.append('rest_brain')

    roires = pd.DataFrame(zip(roi_betas.tolist(), names), columns=["beta", "roi"])
    roires.sort_values('beta', inplace=True, ascending=False)
    ## IF I WANT TO MAKE SEABORN PLOT#####################################
    # labels = roires['roi'].tolist()
    # g = sns.factorplot(x="roi", y="beta", data=roires,  kind="bar", size=6, aspect=1.5)
    # g.set_xticklabels(labels, rotation='vertical')
    # plt.subplots_adjust(bottom=0.35)
    # plt.show()
    ####################################################################
    roi_beta = roires['beta'].tolist()
    labels = roires['roi'].tolist()
    roires.to_csv(os.path.join(stat,score, score + "_" + stat + '_roi-info.csv'), index = None)

    # select data to plot figure
    y = roi_beta
    x = np.arange(len(y))

    # plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.35

    ## the bars
    rects = ax.bar(x, y, width)
    # axes and labels
    plt.xticks(x, labels, rotation='vertical')
    ax.set_xlim(-width,len(x)+width)
    ax.set_ylim(np.min(roi_betas),np.max(roi_betas))
    plt.margins(1)
    plt.subplots_adjust(bottom=0.35)
    plt.ylabel(score + "_" + stat)
    plt.savefig(os.path.join(stat,score,score + "_" + stat + "_"+ title +".png"))
    plt.close()

    return roi_beta, x , labels

def plot_confusion_matrix(cm,  title='Confusion matrix ', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True score', fontsize=18)
    plt.xlabel('Predicted score', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.title("variance",fontsize=18)
    plt.subplots_adjust(bottom=0.30)


def plot_corr(file, score, stat, ind_var, brain_type):

    # seaborn
    sns.set(style="white")

    # import the dataframe
    dt = pd.read_csv(file)

    # Compute the correlation matrix
    corr = dt.corr()

    ### Create the matrix figure with seaborn
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(len(ind_var),len(ind_var)))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=False, ax=ax)
    plt.subplots_adjust(left= 0.30,bottom=0.30)
    plt.savefig(os.path.join(stat,score, "heatmap_" + score + "_" + stat + "_"+ brain_type + ".png"))
    plt.close()

    return corr

def GLM (file, score, stat, ind_var, Level, betas=1):

    # Create pandas dataframe
    df_final= pd.DataFrame(columns=['Score', 'stat', 'beta', 'tvalue', 'pvalue' , 'pval_bonferroni', 'signi_bonferonni', 'Rsquare', 'std'])
    db = pd.read_csv(file)

    # scaler = StandardScaler()
    # for var in ['age', 'age_at_chirurgie']:
    #     db[var] = scaler.fit_transform(db[var])
    #Get rid of rows with null values for given columns
    db = db[db[score].notnull()]

	# Select Variables
    Y = np.array(db[score])
    X = np.array(db[ind_var])


    #Run the GLM
    kf = KFold(Y.shape[0], n_folds=Y.shape[0])
    predictions = []
    rsquares = []
    tas = []
    confus = []
    for train_index, test_index in kf:
        olsmodel = sm.OLS(Y[train_index], X[train_index])
        results = olsmodel.fit()
        pred = np.dot(X[train_index], results.params)
        pred = np.round(pred)
        pred[pred < 0] = 0
        pred[pred > 5] = 5
        ta = np.sum(Y[train_index] == pred) / float(len(Y[train_index]))
        tas.append(ta)
        cm = confusion_matrix(Y[train_index], pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[isnan(cm)] = 0
        if cm.shape[0] == 13:
            confus.append(cm)
        rsquares.append(results.rsquared)
        prediction = np.dot(X[test_index], results.params)
        predictions.append(prediction)
    predictions = np.ravel(predictions)
    confus = np.mean(confus, axis=0)
    plot_confusion_matrix(confus, title='Mean confusion matrix'+stat)
    plt.savefig('Mean confusion matrix_'+stat+"_.png")
    plt.close()

    predictions =  np.round(predictions)
    predictions[predictions<0]=0
    predictions[predictions>5]=5

    cvrsq = 1 - (np.sum((Y - predictions)**2)/np.sum((Y - np.mean(Y))**2))
    # print(stat)
    # print(score)
    # print(cvrsq )
    # rsquares = np.ravel(rsquares)
    # print(rsquares)
    # plt.scatter(predictions, Y)
    # plt.plot([min(Y), max(Y)], [min(Y), max(Y)])
    # plt.xlabel( " time of day prediction for"+" "+ stat)
    # plt.ylabel("time of day score for"+ " " + stat)
    # plt.title("Cross validation Rsquare"+ str(cvrsq))
    # plt.savefig(stat+ ".png")
    # plt.close()
    
    #Compute confusion matrix

    cm = confusion_matrix(Y, predictions)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    total_accuracy = np.sum(Y==predictions)/float(len(Y))
    somme=[]
    diagnonal = np.diagonal(cm)

    for i  in range(len(diagnonal)):
        somme.append((diagnonal[i]/np.sum(cm[:,i]))*100)
    category= np.array(somme)
    NanValue = isnan(category)
    category[NanValue]=0


    # plt.bar(range(6),category)
    # plt.ylabel("accuracy for " + stat)
    # plt.show()


    plt.figure()
    plot_confusion_matrix(cm, title='confusion matrix'+stat)
    plt.savefig('confusion matrix_'+stat+"_.png")
    plt.close()
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix'+stat)
    plt.savefig('Normalized confusion matrix_'+stat+"_.png")
    plt.close()




    model = sm.OLS(Y, X).fit()

    pvals = model.pvalues

    pvals_fwer = multicomp.multipletests(pvals, alpha = 0.05, method = 'fdr_bh')
     
    #Save it into csv file
    df_final.loc[len(df_final)] = [score, stat, model.params, model.tvalues, model.pvalues, pvals_fwer[1], pvals_fwer[0], model.rsquared, model.bse]
    df_final.to_csv(os.path.join(stat,score, score + "_" + stat + "_" + Level + ".csv"))
    
    #check quickly if there is significant data
    for  idx, i in enumerate(model.pvalues):
        if model.pvalues[i] < 0.05:
            print (score+ " " + stat + " "+ Level + ind_var[idx] )
            print (model.pvalues[idx])

    betas_component = model.params[0:betas]

    # ## plot the Betas 
    # Select the variable
    y = model.tvalues

    x = np.array(range(len(ind_var)))
    
     # plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.35

    ## the bars
    rec = ax.bar(x, y, width, color='green')

    # # plot data
    # plt.plot(x, y, linestyle="dashed", marker="o", color="green")
    # plt.bar(x,y,width=0.8, bottom=0.45, hold=None, color = 'r')
    plt.subplots_adjust(bottom=0.45)
    plt.xticks(x, ind_var, rotation='vertical')
    plt.ylabel(score + "_" + stat)
    plt.xlabel("Rsquare %s" % (model.rsquared))

    rects = rec.patches

#xrange(len(model.pvalues))
    # # Now make some labels
    labels = ["p = %f" % i for i in model.pvalues]

    for rect, label in zip(rects, labels):
        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2, height + 2, label, ha='center', va='bottom', weight= 'light', size= 'xx-small')


    plt.savefig(os.path.join(stat, score, score + "_" + stat + "_" + brain_type + "_" + ".png"))
    plt.close()



    return df_final, db, betas_component , pvals, cvrsq, tas,category,total_accuracy

def linearplot(X, Y, df):
    
    # Select x and y 
    x = np.array(df[X])
    y = np.array(df[Y])

     # Run the regression
    beta, beta0, r_value, p_value, std_err = stats.linregress(x, y)
    print("y=%f x + %f, r:%f, r-squared:%f, \np-value:%f, std_err:%f"
    % (beta, beta0, r_value, r_value**2, p_value, std_err))


    # Plot the line
    yhat = beta * x + beta0  # regression line
    plt.plot(x, yhat, 'r-', x, y, 'o')
    plt.xticks(x)
    plt.ylabel(y)
    #plt.show()
    plt.savefig(x+ ""+ y + ".png")
    plt.close()

    return betas_component

# Correlation function to annotate plots in Grids
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate('r = {:.2f}'.format(r), xy=(.1,.9), xycoords=ax.transAxes)
    return ax

def undestand_component(save_to, save_name, stat, betas_component,score):

    # save_to = os.path.join('preprocessed')
    # save_name = (db_name + '_PCA_' + stat + '.pkl')
    pca = joblib.load(os.path.join(save_to, save_name))
    roi_betas = pca.inverse_transform(betas_component)
    order = roi_betas.argsort()[::-1]
    roi_beta, x , ordlabel = plot_hist(roi_betas , stat, score, title="GLM")


    # # Find the contribution of the components on each region
    # labels, names = read_roiLabel(os.path.join('lpba40.label.xml'))
    # names.append('rest_brain')
    # df = pd.DataFrame(columns=[names[x] for x in order.tolist()])
    # list_repetition = [0] * len(betas_component)
    # for idx, i in enumerate(range(len(list_repetition))):
    #     new_list = list(list_repetition)
    #     new_list[i] = 1
    #     comp_betas = pca.inverse_transform(new_list)
    #     # composante = [idx]
    #     df.loc[len(df)] = comp_betas[order]

    # plt.imshow(df, interpolation='nearest')
    # plt.xticks(range(len(df.columns.tolist())), df.columns.tolist(), rotation='vertical')
    # plt.yticks(range(len(list_repetition)))
    # plt.set_cmap('spectral')
    # plt.colorbar()
    # plt.savefig(os.path.join(stat, stat + "_component_contribution_for_each_region_.png"))


    save_name_rd = (db_name + '_RD_'+ stat + '.pkl')
    # pca = joblib.load(os.path.join(save_to, save_name))
    # pca = joblib.load(os.path.join(save_to, save_name))

    RD = joblib.load(os.path.join(save_to, save_name_rd))
    pca_compo_roi = pca.transform(RD)


    # Turn RD and components into DataFrames
    comp_columns = ['component-%i' % i for i in range(pca_compo_roi.shape[1])]

    compdf = pd.DataFrame(pca_compo_roi, columns=comp_columns)

    labels, names = read_roiLabel(os.path.join('lpba40.label.xml'))

    # roi_columns = ['roi%i' % i for i in range(RD.shape[1])]
    roi_columns = names
    names.append('rest_brain')
    roidf = pd.DataFrame(RD, columns=roi_columns)


    # Plot components bivariate relationships
    g = sns.PairGrid(compdf)
    g = g.map_diag(sns.kdeplot)
    g = g.map_upper(plt.scatter)
    g = g.map_lower(sns.kdeplot)
    g = g.map_lower(corrfunc)
    #plt.show()
    plt.savefig(os.path.join(stat,"PCA_compo_capturing_features_between_subject_" + stat + ".png"))
    plt.close()

   # Plot regression of component on each ROI
    # n_parts = 4
    # for  idx, i in enumerate(range(n_parts)):
    #     part = 56/n_parts
    #     selecroi = names[part*i: part*(i+1)]
    #     #selecroi = roi_columns = ['roi%i' % i for i in range(part*i, part*(i+1))]
    #     combineddf = pd.concat([compdf, roidf[selecroi]], axis=1)
    #     g = sns.PairGrid(combineddf, x_vars=selecroi, y_vars=comp_columns)
    #     g = g.map(sns.regplot)
    #     g = g.map(corrfunc)
    #     plt.savefig(os.path.join(stat,"PCA_compo_roi_density_part_" + str(idx) + stat +  ".png"))
    #     plt.close()

    # Plot heatmap of component and ROI correlations
    n_parts = 1
    for idx, i  in enumerate(range(n_parts)):
        part = 56/n_parts
        selecroi = ordlabel[part*i: part*(i+1) + 1]
        # selecroi = roi_columns = ['roi%i' % i for i in range(part*i, part*(i+1))]
        combineddf = pd.concat([compdf, roidf[selecroi]], axis=1)

        sns.heatmap(combineddf.corr()[selecroi].loc[comp_columns], annot=False )
        plt.subplots_adjust(bottom=0.40)
        plt.show()
        plt.savefig(os.path.join(stat,"PCA_compo_roi_heatmap_"+ str(idx) + stat + ".png"))
        plt.close()
    # # Plot heatmap of component and ROI correlations
    # n_parts = 2
    # for idx, i  in enumerate(range(n_parts)):
    #     part = 56/n_parts
    #     selecroi = names[part*i: part*(i+1)]
    #     # selecroi = roi_columns = ['roi%i' % i for i in range(part*i, part*(i+1))]
    #     combineddf = pd.concat([compdf, roidf[selecroi]], axis=1)
    #     sns.heatmap(combineddf.corr()[selecroi].loc[comp_columns], annot=False )
    #     plt.subplots_adjust(bottom=0.40)
    #     plt.savefig(os.path.join(stat,"PCA_compo_roi_heatmap_"+ str(idx) + stat + ".png"))
    #     plt.close()

    return None

if __name__ == '__main__':

    db_name = "hippomuse_elotin"
    Path = "preprocessed_hippomuse"
    Files = glob.glob(os.path.join(Path, "*.csv"))
    Scores = ['reconnaissanceImageP1', 'reconnaissanceOdeurP1',
               'musique_P1', 'musique_P2', 'when_P1', 'heure_P1',
               'repereJournalier_P1', 'what_P1', 'episodicite_P1',
               'when_P2', 'heure_P2', 'repereJournalier_P2', 'what_P2',
               'episodicite_P2', 'delta_when_P2', 'delta_heure_P2',
               'delta_repereJournalier_P2', 'delta_what_P2',
               'delta_episodicite_P2']
    df_Rsqures = pd.DataFrame(columns= ["Statistic","Cross validation Rsquare", "Rsquare distribution", "Analysis level"])
    Statistic = []
    CVRsq = []
    Rsquares = []
    Level_analysis = []
    accuracy =[]
    list_total_accuracy=[]


    #df_corr_var = pd.DataFrame(columns=["component_01","component_02","component_03","component_04","component_05","component_06", "component_07","chimiotherapie","chirurgie_delay_bilan","age_at_chirurgie" , "DVP", "VCS", "age"] )
    # scores = ['aha', 'ehe']
    # # for f, score in [(f, score) for f in Files for score in scores]:
    # for f, score in itertools.product(Files, scores):
    #     stat = f.split('_')[3].split('.')[0]

    for f in Files:
        stat = f.split('_')[4].split('.')[0]
        #ind_var = ["radiotherapie", "chimiotherapie", "chirurgie_delay_bilan", "age_at_chirurgie",  "DVP", "VCS", "age_diff_bilan"]
        # save_to = os.path.join('preprocessed_hippomuse')
        # save_name = (db_name + '_PCA_' + stat + '.pkl')
        # save_name_rd = (db_name + '_RD_'+ stat + '.pkl')
        # undestand_component(save_to, save_name, stat, betas_component)

        for score in Scores[9:10]:
        #     delta =  score.split("_")[0]
        #     if delta == "delta":
        #         p1 = score.split("_")[1] + "P1"
        #     else:
        #         continue


            if stat == "presence":
                brain_type = "presence"
                # ind_var = ["radiotherapie", "chimiotherapie", "DVP","age", "age_at_chirurgie"]
                # df_final, db, betas_component, pvals , cvrsq, rsquares,category,total_accuracy = GLM (f, score, stat, ind_var, brain_type)



                # corr = plot_corr(f, score, stat, ind_var, brain_type)


                patient_effect = ["groupe","age"]
                df_final, db,betas_component, pvals , cvrsq, rsquares,category,total_accuracy = GLM (f, score, stat, patient_effect, brain_type)


                g= sns.lmplot(x="age", y=score, data=db, hue="groupe", markers=["o", "x"], palette=['b','g'])
                plt.tick_params(labelsize=30)
                g.set_ylabels("When at recognition",size =30,color="black")
                g.set_xlabels("Age",size =30,color="black")    
                plt.show()
                plt.savefig("Patient_effect_" + score + "_" + stat + "_" + ".png")
                plt.close()                



            # else:

            #     brain_type = f.split('_')[5].split('.')[0]
            #     if brain_type == "atlas":

            #         if stat == "max":
            #             ind_var = ["component_01","component_02","component_03","chimiotherapie", "DVP","age_at_chirurgie","age"]                
            #             df_final, db, betas_component, pvals , cvrsq, rsquares,category,total_accuracy= GLM (f, score, stat, ind_var, brain_type, betas=3)
            #             Statistic.append(stat)
            #             CVRsq.append(cvrsq)
            #             Rsquares.append(rsquares)
            #             Level_analysis.append(brain_type)
            #             accuracy.append(category)
            #             list_total_accuracy.append(total_accuracy)

            #             # save_to = os.path.join('preprocessed_hippomuse')
            #             # save_name = (db_name + '_PCA_' + stat + '.pkl')
            #             # save_name_rd = (db_name + '_RD_'+ stat + '.pkl')
            #             # undestand_component(save_to, save_name, stat, betas_component, score)



            #             # # Find the contribution of the components on each region
            #             # labels, names = read_roiLabel(os.path.join('lpba40.label.xml'))
            #             # names.append('rest_brain')
            #             # df = pd.DataFrame(columns=[names[x] for x in order.tolist()])
            #             # list_repetition = [0] * len(betas_component)
            #             # for idx, i in enumerate(range(len(list_repetition))):
            #             #     new_list = list(list_repetition)
            #             #     new_list[i] = 1
            #             #     comp_betas = pca.inverse_transform(new_list)
            #             #     # composante = [idx]
            #             #     df.loc[len(df)] = comp_betas[order]
            #             # plt.imshow(df, interpolation='nearest')
            #             # plt.xticks(range(len(df.columns.tolist())), df.columns.tolist(), rotation='vertical')
            #             # plt.yticks(range(len(list_repetition)))
            #             # plt.set_cmap('spectral')
            #             # plt.colorbar()
            #             # plt.show()
            #               # roires = pd.DataFrame(zip(roi_betas.tolist(), names, composante*len(names)), columns=["beta", "roi", "component"])
            #                 # roires.sort_values('beta', inplace=True, ascending=False)
                       
            #                 # y, x = plot_hist(roi_betas, stat, score, component= idx, title="Components_%i"%(idx))


            #             corr = plot_corr(f, score, stat, ind_var, brain_type)

            #         if stat == "sum":
            #             ind_var = ["component_01","component_02","component_03","component_04","component_05","chimiotherapie", "DVP","age_at_chirurgie", "age"]                
            #             df_final, db, betas_component,pvals, cvrsq, rsquares,category ,total_accuracy= GLM (f, score, stat, ind_var, brain_type, betas=5)
            #             Statistic.append(stat)
            #             CVRsq.append(cvrsq)
            #             Rsquares.append(rsquares)
            #             Level_analysis.append(brain_type)
            #             accuracy.append(category)
            #             list_total_accuracy.append(total_accuracy)

            #             # save_to = os.path.join('preprocessed_hippomuse')
            #             # save_name = (db_name + '_PCA_' + stat + '.pkl')
            #             # save_name_rd = (db_name + '_RD_'+ stat + '.pkl')
            #             # undestand_component(save_to, save_name, stat, betas_component,score)


            #             corr = plot_corr(f, score, stat, ind_var, brain_type)

            #         if stat == "min":
            #             ind_var = ["component_01","component_02","component_03","component_04","chimiotherapie", "DVP","age_at_chirurgie","age"]                
            #             df_final, db, betas_component,pvals, cvrsq, rsquares,category,total_accuracy = GLM (f, score, stat, ind_var, brain_type, betas=4)
            #             Statistic.append(stat)
            #             CVRsq.append(cvrsq)
            #             Rsquares.append(rsquares)
            #             Level_analysis.append(brain_type)
            #             accuracy.append(category)
            #             list_total_accuracy.append(total_accuracy)

            #             # save_to = os.path.join('preprocessed_hippomuse')
            #             # save_name = (db_name + '_PCA_' + stat + '.pkl')
            #             # save_name_rd = (db_name + '_RD_'+ stat + '.pkl')
            #             # undestand_component(save_to, save_name, stat, betas_component,score)
                        

            #             corr = plot_corr(f, score, stat, ind_var, brain_type)


            #         if stat == 'var':
            #             ind_var =["component_01","component_02","component_03","component_04","component_05","component_06", "component_07","chimiotherapie", "DVP","age_at_chirurgie" , "age"]
            #             df_final, db , betas_component,pvals, cvrsq, rsquares,category,total_accuracy= GLM (f, score, stat, ind_var, brain_type, betas=7)
            #             Statistic.append(stat)
            #             CVRsq.append(cvrsq)
            #             Rsquares.append(rsquares)
            #             Level_analysis.append(brain_type)
            #             accuracy.append(category)
            #             list_total_accuracy.append(total_accuracy)

            #             # save_to = os.path.join('preprocessed_hippomuse')
            #             # save_name = (db_name + '_PCA_' + stat + '.pkl')
            #             # save_name_rd = (db_name + '_RD_'+ stat + '.pkl')
            #             # undestand_component(save_to, save_name, stat, betas_component,score)

            #             corr = plot_corr(f, score, stat, ind_var, brain_type)

            #         if stat == 'mean':
            #             ind_var = ["component_01","component_02","component_03","chimiotherapie", "DVP","age_at_chirurgie", "age"]                
            #             df_final, db, betas_component, pvals, cvrsq, rsquares ,category,total_accuracy= GLM (f, score, stat, ind_var, brain_type, betas=3)
            #             Statistic.append(stat)
            #             CVRsq.append(cvrsq)
            #             Rsquares.append(rsquares)
            #             Level_analysis.append(brain_type)
            #             accuracy.append(category)
            #             list_total_accuracy.append(total_accuracy)


            #             # save_to = os.path.join('preprocessed_hippomuse')
            #             # save_name = (db_name + '_PCA_' + stat + '.pkl')
            #             # save_name_rd = (db_name + '_RD_'+ stat + '.pkl')
            #             # undestand_component(save_to, save_name, stat, betas_component,score)

            #             corr = plot_corr(f, score, stat, ind_var, brain_type)

            #         if stat == 'median':
            #             ind_var = ["component_01","component_02","component_03", "chimiotherapie", "DVP","age_at_chirurgie","age"]                
            #             df_final, db, betas_component ,pvals, cvrsq, rsquares,category ,total_accuracy= GLM (f, score, stat, ind_var, brain_type, betas=3)
            #             Statistic.append(stat)
            #             CVRsq.append(cvrsq)
            #             Rsquares.append(rsquares)
            #             Level_analysis.append(brain_type)
            #             accuracy.append(category)
            #             list_total_accuracy.append(total_accuracy)

            #             # save_to = os.path.join('preprocessed_hippomuse')
            #             # save_name = (db_name + '_PCA_' + stat + '.pkl')
            #             # save_name_rd = (db_name + '_RD_'+ stat + '.pkl')
            #             # undestand_component(save_to, save_name, stat, betas_component,score)

            #             corr = plot_corr(f, score, stat, ind_var, brain_type)

            #     # else:

                #     ind_var = ["radiotherapie", "chimiotherapie","age_at_chirurgie" ,"age","DVP"]
                #     df_final, db, betas_component,pvals , cvrsq, rsquares,category= GLM (f, score, stat, ind_var, brain_type)
                #     # Statistic.append(stat)
                #     # CVRsq.append(cvrsq)
                #     # Rsquares.append(rsquares)
                #     # Level_analysis.append(brain_type)

                    # corr = plot_corr(f, score, stat, ind_var, brain_type)
    df_Rsqures["Statistic"] = Statistic
    df_Rsqures["Cross validation Rsquare"]= CVRsq
    df_Rsqures["Rsquare distribution"] = Rsquares
    df_Rsqures["Analysis level"] = Level_analysis

    # # Create plot for R sqaures
    # fig = plt.figure(1, figsize=(9, 6))
    # # Create an axes instance
    # ax = fig.add_subplot(111)
    # Create the boxplot
    A = np.vstack(Rsquares)
    Z = pd.DataFrame(A)
    Z = Z.T
    Z.columns = ['min', 'median', 'mean', 'max', 'sum', 'variance']
    #Z.T.boxplot(vert=False)

    fig, axes = plt.subplots(1, figsize=(8,4))
    df_tot_accu = pd.read_csv("Acurracy_hour.csv")
    g = sns.violinplot(data=Z, orient="h", palette="Set2")

    for tick in g.xaxis.get_major_ticks():
        tick.label.set_fontsize(30)
    for tick in g.yaxis.get_major_ticks():
        tick.label.set_fontsize(30)
        # tick.label.set_rotation('vertical')
    # g.legend(title='Statistic', loc='upper left')
    plt.xlim(0.15,0.50)
    plt.show()
    # o_patch = mpatches.Patch(color= "white",label='Accuracy distribution (train set)',markersize=15)
    # a_patch = mpatches.Patch(color= "white",label='0-min')
    # b_patch = mpatches.Patch(color= "white",label='1-median')
    # c_patch = mpatches.Patch(color= "white",label='2-mean')
    # d_patch = mpatches.Patch(color= "white",label='3-max')
    # e_patch = mpatches.Patch(color= "white",label='4-sum')
    # f_patch = mpatches.Patch(color= "white",label='4-variance')
    # g_patch = mpatches.Patch(color= "black",label='total accuracy (test set')
    # plt.legend(handles=[o_patch, a_patch, b_patch, c_patch, d_patch, e_patch, f_patch,g_patch])
    # plt.legend(prop={'size':20})



    #sns.swarmplot(data=df_tot_accu[df_tot_accu.Score=="total"], orient="h", color=".25")
    #sns.factorplot(data=plo.T, orient="h", palette="Set2",ax=axes[1])



        # for f in Files:

        #     if f.split('_')[3].split('.')[0] == "presence":
        #         print (f)
        #         df = pd.read_csv(f)
        #         score = f.split('_')[2]
        #         ind_var = ["radiotherapie", "chimiotherapie", "chirurgie_delay_bilan", "age_at_chirurgie",  "DVP", "VCS", "age_diff_bilan"]

        #          # plot linear plot
        #         for idx , i in enumerate(ind_var):
        #             sns.lmplot(x=ind_var[idx], y = "delta_"+score, data=df )
        #             plt.savefig("plot" + "_"+ ind_var[idx]+ "_"+ "delta_"+score + ".png")
        #             plt.close()
        #     else:
        #         continue
