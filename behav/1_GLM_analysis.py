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
from sklearn.preprocessing import StandardScaler

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
    roi_beta = roires['beta'].tolist()
    labels = roires['roi'].tolist()
    roires.to_csv(os.path.join(score + "_" + stat + '_roi-info.csv'), index = None)

    # select data to plot figure
    y = roi_beta
    x = np.arange(len(y))

    # plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.35

    ## the bars
    rects = ax.bar(x, y, width, color='blue')
    
    # axes and labels
    plt.xticks(x, labels, rotation='vertical')
    ax.set_xlim(-width,len(x)+width)
    ax.set_ylim(np.min(roi_betas),np.max(roi_betas))
    plt.margins(1)
    plt.subplots_adjust(bottom=0.35)
    plt.ylabel(score + "_" + stat)
    plt.savefig(score + "_" + stat + "_"+ title +".png")
    plt.close()

    return roi_beta, x 

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
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, ax=ax)
    plt.subplots_adjust(left= 0.20,bottom=0.20)
    plt.savefig(score+ "_" + stat + "_"+ brain_type + ".png")
    plt.close()

    return corr

def GLM (file, score, stat, ind_var, Level, betas=1):

    # Create pandas dataframe
    df_final= pd.DataFrame(columns=['Score', 'stat', 'beta', 'tvalue', 'pvalue' , 'pval_bonferroni', 'signi_bonferonni', 'Rsquare', 'std'])
    db = pd.read_csv(file)

    # Get rid of rows with null values for given columns
    db = db[db["delta_" + score].notnull()]

	# Select Variables
    Y = np.array(db["delta_" + score])
    X = np.array(db[ind_var])

    #Run the GLM
    model = sm.OLS(Y, X).fit()
    pvals = model.pvalues
    pvals_fwer = multicomp.multipletests(pvals, alpha = 0.05, method = 'fdr_bh')
     
    #Save it into csv file
    df_final.loc[len(df_final)] = [score, stat, model.params, model.tvalues, model.pvalues, pvals_fwer[1], pvals_fwer[0], model.rsquared, model.bse]
    df_final.to_csv(score + "_" + stat + "_" + Level + ".csv")
    
    # check quickly if there is significant data
    for  idx, i in enumerate(model.pvalues):
        if model.pvalues[i] < 0.05:
            print (score+ " " + stat + " "+ Level + ind_var[idx] )
            print (model.pvalues[idx])

    betas_component = model.tvalues[0:betas]

    # ## plot the Betas 
    # Select the variable
    y = model.tvalues
    x = np.array(range(len(ind_var)))

    # plot data
    plt.plot(x, y, linestyle="dashed", marker="o", color="green")
    plt.xticks(x, ind_var)
    plt.ylabel(score + "_" + stat)
    plt.xlabel("Rsquare %s" % (model.rsquared))
    plt.savefig(score + "_" + stat + "_" + ".png")
    plt.close()

    return df_final, db, betas_component

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

if __name__ == '__main__':
    
    db_name = "all_patients"
    Path = "preprocessed"
    Files = glob.glob(os.path.join(Path, "*.csv"))


    # scores = ['aha', 'ehe']
    # # for f, score in [(f, score) for f in Files for score in scores]:
    # for f, score in itertools.product(Files, scores):
    #     stat = f.split('_')[3].split('.')[0]

    for f in Files:
        score = f.split('_')[2]
        stat = f.split('_')[3].split('.')[0]

	 #ind_var = ["radiotherapie", "chimiotherapie", "chirurgie_delay_bilan", "age_at_chirurgie",  "DVP", "VCS", "age_diff_bilan"]
        if stat == "presence":
            brain_type = "_"
            ind_var = ["radiotherapie", "chimiotherapie", "chirurgie_delay_bilan","age_at_chirurgie", "age_diff_bilan" , "DVP", "VCS"]
            df_final, db, betas_component = GLM (f, score, stat, ind_var, brain_type)
            corr = plot_corr(f, score, stat, ind_var, brain_type)

    

        else:

            brain_type = f.split('_')[4].split('.')[0]
            if brain_type == "atlas":
                if stat == "max":
                    ind_var = ["component_01","component_02","component_03", "component_04","chimiotherapie","chirurgie_delay_bilan","age_at_chirurgie", "age_diff_bilan" , "DVP", "VCS"]                
                    df_final, db, betas_component = GLM (f, score, stat, ind_var, brain_type, betas=4)



                    # save_to = os.path.join('preprocessed')
                    # save_name = (db_name + '_PCA_' + stat + '.pkl')
                    # pca = joblib.load(os.path.join(save_to, save_name))
                    # roi_betas = pca.inverse_transform(betas_component)
                    # order = roi_betas.argsort()[::-1]
                    # y, x = plot_hist(roi_betas , stat, score, title="GLM")



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
                    # plt.show()

                    save_to = os.path.join('preprocessed')
                    save_name_rd = (db_name + '_RD_'+ stat + '.pkl')
                    save_name = (db_name + '_PCA_' + stat + '.pkl')
                    # pca = joblib.load(os.path.join(save_to, save_name))
                    pca = joblib.load(os.path.join(save_to, save_name))
                    RD = joblib.load(os.path.join(save_to, save_name_rd))
                    pca_compo_roi = pca.transform(RD)

                    # Correlation function to annotate plots in Grids
                    def corrfunc(x, y, **kws):
                        r, _ = stats.pearsonr(x, y)
                        ax = plt.gca()
                        ax.annotate('r = {:.2f}'.format(r), xy=(.1,.9), xycoords=ax.transAxes)

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
                    plt.savefig("PCA_compo_capturing_features_between_subject" + stat + ".png")
                    plt.close()

                    # Plot regression of component on each ROI
                    n_parts = 4
                    for i in range(n_parts):
                        part = 56/n_parts
                        selecroi = names[part*i: part*(i+1)]
                        # selecroi = roi_columns = ['roi%i' % i for i in range(part*i, part*(i+1))]
                        combineddf = pd.concat([compdf, roidf[selecroi]], axis=1)
                        g = sns.PairGrid(combineddf, x_vars=roi_columns, y_vars=comp_columns)
                        g = g.map(sns.regplot)
                        g = g.map(corrfunc)
                        plt.savefig("PCA_compo_roi_density" + stat + ".png")
                        plt.close()


                    # Plot heatmap of component and ROI correlations
                    n_parts = 2
                    for i in range(n_parts):
                        part = 56/n_parts
                        selecroi = names[part*i: part*(i+1)]
                        # selecroi = roi_columns = ['roi%i' % i for i in range(part*i, part*(i+1))]
                        combineddf = pd.concat([compdf, roidf[selecroi]], axis=1)
                        sns.heatmap(combineddf.corr()[names].loc[comp_columns])
                        plt.subplots_adjust(bottom=0.20)
                        plt.savefig("PCA_compo_roi_heatmap" + stat + ".png")
                        plt.close()

                    import pdb; pdb.set_trace()  # breakpoint 7f2f5d2a //

                        # roires = pd.DataFrame(zip(roi_betas.tolist(), names, composante*len(names)), columns=["beta", "roi", "component"])
                        # roires.sort_values('beta', inplace=True, ascending=False)
                   
                        # y, x = plot_hist(roi_betas, stat, score, component= idx, title="Components_%i"%(idx))




                   

                    corr = plot_corr(f, score, stat, ind_var, brain_type)

                if stat == "sum":
                    ind_var = ["component_01","component_02","component_03","component_04","component_05","component_06","chimiotherapie","chirurgie_delay_bilan","age_at_chirurgie", "age_diff_bilan" , "DVP", "VCS"]                
                    df_final, db, betas_component = GLM (f, score, stat, ind_var, brain_type, betas=6)

                    save_to = os.path.join('preprocessed')
                    save_name = (db_name + '_PCA_' + stat + '.pkl')
                    pca = joblib.load(os.path.join(save_to, save_name))
                    roi_betas = pca.inverse_transform(betas_component)
                    y, x = plot_hist(roi_betas, stat, score,title="GLM")


                    corr = plot_corr(f, score, stat, ind_var, brain_type)

                if stat == "min":
                    ind_var = ["component_01","component_02","component_03","component_04","component_05","component_06","chimiotherapie","chirurgie_delay_bilan","age_at_chirurgie", "age_diff_bilan" , "DVP", "VCS"]                
                    df_final, db, betas_component = GLM (f, score, stat, ind_var, brain_type, betas=6)
                    
                    save_to = os.path.join('preprocessed')
                    save_name = (db_name + '_PCA_' + stat + '.pkl')
                    pca = joblib.load(os.path.join(save_to, save_name))
                    roi_betas = pca.inverse_transform(betas_component)
                    y, x = plot_hist(roi_betas, stat, score, title="GLM")
                    

                    corr = plot_corr(f, score, stat, ind_var, brain_type)


                if stat == 'var':
                    ind_var =["component_01","component_02","component_03","component_04","component_05","component_06", "component_07","component_08","chimiotherapie","chirurgie_delay_bilan","age_at_chirurgie", "age_diff_bilan" , "DVP", "VCS"]                
                    df_final, db , betas_component= GLM (f, score, stat, ind_var, brain_type, betas=8)
                    
                    save_to = os.path.join('preprocessed')
                    save_name = (db_name + '_PCA_' + stat + '.pkl')
                    pca = joblib.load(os.path.join(save_to, save_name))
                    roi_betas = pca.inverse_transform(betas_component)
                    y, x = plot_hist(roi_betas, stat, score, title="GLM")

                    corr = plot_corr(f, score, stat, ind_var, brain_type)

                if stat == 'mean':
                    ind_var = ["component_01","component_02","component_03","chimiotherapie","chirurgie_delay_bilan","age_at_chirurgie", "age_diff_bilan" , "DVP", "VCS"]                
                    df_final, db, betas_component = GLM (f, score, stat, ind_var, brain_type, betas=3) 

                    save_to = os.path.join('preprocessed')
                    save_name = (db_name + '_PCA_' + stat + '.pkl')
                    pca = joblib.load(os.path.join(save_to, save_name))
                    roi_betas = pca.inverse_transform(betas_component)
                    y, x = plot_hist(roi_betas, stat, score, title="GLM")

                    corr = plot_corr(f, score, stat, ind_var, brain_type)

                if stat == 'median':
                    ind_var = ["component_01","component_02","component_03", "chimiotherapie","chirurgie_delay_bilan","age_at_chirurgie", "age_diff_bilan" , "DVP", "VCS"]                
                    df_final, db, betas_component = GLM (f, score, stat, ind_var, brain_type, betas=3)

                    save_to = os.path.join('preprocessed')
                    save_name = (db_name + '_PCA_' + stat + '.pkl')
                    pca = joblib.load(os.path.join(save_to, save_name))
                    roi_betas = pca.inverse_transform(betas_component)
                    y, x = plot_hist(roi_betas, stat, score, title="GLM")

                    corr = plot_corr(f, score, stat, ind_var, brain_type)


            if brain_type  == "brain":
                ind_var = ["radiotherapie", "chimiotherapie", "chirurgie_delay_bilan","age_at_chirurgie", "age_diff_bilan" , "DVP", "VCS"]
                df_final, db, betas_component= GLM (f, score, stat, ind_var, brain_type)
                corr = plot_corr(f, score, stat, ind_var, brain_type)


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

