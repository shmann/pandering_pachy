''' Run tSNE on book factors with genre labels

Usage:
    $ python genre_tsne.py filename_book_factors_top_genres.csv filename_genre_id_map.csv perplexity train_pct print_pct
    
    Note: takes integer percentages only
    
E.g.

python genre_tsne.py ~/rank_50_reg_0.05_pct_100_als_model_book_factors_top_genres_merged.csv ~/genre_id_map.csv 50 5 100

'''

import sys
import pandas as pd
import numpy as np
from tsnecuda import TSNE
import matplotlib.pyplot as plt
from sklearn.utils import resample, shuffle

def downsample_by_pct(X,l,pct):
    
    X_ds,l_ds = resample(X,l,n_samples=int(round(len(X)*pct/100)),random_state=1)
    
    print("{} total samples, taking {} ({}%)".format(len(l),len(l_ds),pct))
    
    return X_ds, l_ds

def downsample_to_min(X,l,genre_id_map):
    
    X_ds=pd.DataFrame()
    l_ds=[]
    
    for g in genre_id_map.genre_id:
        
        ix = np.where(l == g)
        
        X_g=X.loc[ix]
        l_g=l[ix]
        
        min_samples=min(np.bincount(l))
        
        X_g_ds,l_g_ds=resample(X_g,l_g,n_samples=min_samples,random_state=1)
        
        X_ds = X_ds.append(X_g_ds)
        l_ds.extend(l_g_ds)
        
    X_ds, l_ds = shuffle(X_ds, l_ds)
    
    print("downsampled {} samples to genre with fewest samples. {:.1f}% samples remain.".format(len(l),100*(len(l_ds)/len(l))))
    
    return X_ds, np.asarray(l_ds)

def save_tSNE_split(transformed, l, genre_id_map, colors, downsample=False, pct=100):
    
    if pct:
        n_samples=int(round(len(transformed)*pct/100))
        print("{} total samples, taking {} ({}%)".format(len(transformed),n_samples,pct))
        transformed,l=resample(transformed,l,n_samples=n_samples,random_state=1)
    
    legend=genre_id_map.set_index("genre_id").to_dict().get("genre")
    
    for g in genre_id_map.genre_id:
    
        fig,ax=plt.subplots(figsize=(15,10))
    
        ix = np.where(l == g)
        
        x=transformed[ix,0][0]
        y=transformed[ix,1][0]
        
        if downsample:
            min_samples=min(np.bincount(l))
            print("\t{} has {} samples, printing {}".format(legend[g],len(x),min_samples))
            x,y=resample(x,y,n_samples=min_samples,random_state=1)
        
        ax.scatter(x, y, label=legend[g], s=15, alpha=1, c=colors[g], antialiased=False)
        ax.set(xlim=(-25, 25), ylim=(-25, 25))
        plt.savefig(legend[g]+".png")
        #plt.show()
        plt.close()

def print_tSNE(transformed, l, genre_id_map, colors, downsample=False, pct=100):
    
    if pct:
        n_samples=int(round(len(transformed)*pct/100))
        #print("{} total samples, taking {} ({}%)".format(len(transformed),n_samples,pct))
        transformed,l=resample(transformed,l,n_samples=n_samples,random_state=1)
    
    legend=genre_id_map.set_index("genre_id").to_dict().get("genre")
    
    fig,ax=plt.subplots(figsize=(15,10))
    
    for g in genre_id_map.genre_id:
    
        ix = np.where(l == g)
        
        x=transformed[ix,0][0]
        y=transformed[ix,1][0]
        
        if downsample:
            min_samples=min(np.bincount(l))
            #print("{} has {} samples, printing {}".format(legend[g],len(x),min_samples))
            x,y=resample(x,y,n_samples=min_samples,random_state=1)
        
        ax.scatter(x, y, label=legend[g], s=15, alpha=.2, c=colors[g] ,edgecolors="face")
    
    leg = ax.legend(markerscale=1)
    
    ax.set(xlim=(-25, 25), ylim=(-25, 25))
    
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    
    plt.savefig("combined.png")
    #plt.show()
    plt.close()
    

def main(book_factors_top_genres, genre_id_map, perplexity, train_pct, print_pct):
    
    n_features = pd.read_csv(book_factors_top_genres,sep=",",header=0,nrows=1).shape[1] -2

    data_header=["book_id","genre_id"]

    for x in range(n_features):
        data_header.append("x"+str(x))

    print("reading data...")
    book_factors_top_genres = pd.read_csv(book_factors_top_genres,sep=",",header=0,names=data_header)

    genre_id_map = pd.read_csv(genre_id_map,sep=",",header=None,names=["genre_id","genre"])

    book_factors_top_genres=book_factors_top_genres.join(genre_id_map.set_index("genre_id"),on="genre_id",how="left")

    X = book_factors_top_genres.drop(["book_id","genre_id","genre"],axis=1)

    l = np.asarray(book_factors_top_genres.genre_id)

    tSNE = TSNE(
        n_components=2
        , perplexity=perplexity #sklearn 30.0 default 50.0
    )

    #rename for downsampling
    X_ds,l_ds=X,l

    #not used. for evening out the genres before running tSNE. did not improve results.
    #X_ds,l_ds=downsample_to_min(X_ds,l_ds,genre_id_map)
    
    #downsampling does, however, improve results
    X_ds,l_ds=downsample_by_pct(X_ds,l_ds,pct=train_pct)

    print("training tSNE...")
    transformed=tSNE.fit_transform(X_ds)

    #randomly select colors. some colors are easier to see than others.
    colors=shuffle(["darkgreen","darkblue","maroon","red","gold","lawngreen","aqua","pink","brown","salmon"])
    
    print("saving figures")
    
    save_tSNE_split(transformed, l_ds, genre_id_map, colors, downsample=True, pct=print_pct)
    print_tSNE(transformed, l_ds, genre_id_map, colors, downsample=True, pct=print_pct)


if __name__=="__main__":
    
    book_factors_top_genres=sys.argv[1]
    genre_id_map=sys.argv[2]
    perplexity=float(sys.argv[3])
    train_pct=int(sys.argv[4])
    print_pct=int(sys.argv[5])
    
    main(book_factors_top_genres, genre_id_map, perplexity, train_pct, print_pct)

