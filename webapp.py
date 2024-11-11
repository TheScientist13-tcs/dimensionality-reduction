import streamlit as st
import numpy as np
import plotly.express as px
from sklearn import datasets
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## update


def label(x):
    if x == 0:
        return "setosa"
    elif x == 1:
        return "versicolor"
    return "virginica"


def main():
    # Set the title of the app
    st.set_page_config(page_title="PCA", layout="wide")
    st.title("Principal Component Analysis Demo")
    st.markdown(
        "##### By: Dharyll Prince M. Abellana | Assistant Professor of Computer Science | University of the Philippines Cebu"
    )

    st.write("This app uses the IRIS dataset for demonstration purposes only")

    st.write(
        ":red[NOTE: You can interact with the figure by clicking and dragging your mouse/trackpad]"
    )

    # Sidebar for user input
    st.sidebar.header("Click this button to compute the PCA")

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    y = list(map(label, y))
    feature_names = iris.feature_names

    scaler = StandardScaler()
    scaler.fit(X)
    X_trans = scaler.transform(X)

    df = pd.DataFrame(data=X_trans, columns=feature_names)
    df["target"] = y

    with st.sidebar:
        compute_pca = st.button("Compute PCA")

    if not compute_pca:
        fig = px.scatter_3d(
            df,
            x=feature_names[0],
            y=feature_names[1],
            size=X[:, -1],
            z=feature_names[2],
            color="target",
            width=1000,
            height=700,
        )

        fig.update_layout(
            scene=dict(xaxis=dict(title="Sepal Length", range=[-5, 5], nticks=5))
        )
        fig.update_layout(
            scene=dict(yaxis=dict(title="Sepal Width", range=[-5, 5], nticks=5))
        )
        fig.update_layout(
            scene=dict(zaxis=dict(title="Petal Length", range=[-5, 5], nticks=5))
        )
        fig.update_layout(
            dragmode="drawopenpath",
            scene=dict(
                aspectmode="auto",  # Set aspect mode to manual
                aspectratio=dict(x=1, y=1, z=1),  # Set equal ratios for all axes
            ),
        )

        st.plotly_chart(fig)

    if compute_pca:
        fig = px.scatter_3d(
            df,
            x=feature_names[0],
            y=feature_names[1],
            size=X[:, -1],
            z=feature_names[2],
            color="target",
            width=1000,
            height=700,
        )

        fig.update_layout(
            scene=dict(xaxis=dict(title="Sepal Length", range=[-5, 5], nticks=5))
        )
        fig.update_layout(
            scene=dict(yaxis=dict(title="Sepal Width", range=[-5, 5], nticks=5))
        )
        fig.update_layout(
            scene=dict(zaxis=dict(title="Petal Length", range=[-5, 5], nticks=5))
        )
        fig.update_layout(
            dragmode="drawopenpath",
            scene=dict(
                aspectmode="manual",  # Set aspect mode to manual
                aspectratio=dict(x=1, y=1, z=1),  # Set equal ratios for all axes
            ),
        )

        pca = PCA(n_components=2).fit(X_trans)

        X_pca = pca.transform(X_trans)

        df_pca = pd.DataFrame(data=X_pca, columns=["PC 1", "PC 2"])
        df_pca["target"] = y

        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_
        sc_val = 5

        sc_val_1 = eigenvalues[0]
        fig.add_trace(
            go.Scatter3d(
                x=[-sc_val_1 * eigenvectors[0][0], sc_val_1 * eigenvectors[0][0]],
                y=[-sc_val_1 * eigenvectors[0][1], sc_val_1 * eigenvectors[0][1]],
                z=[-sc_val_1 * eigenvectors[0][2], sc_val_1 * eigenvectors[0][2]],
                line=dict(color="red", width=1),
                mode="lines",
                name="PC 1",
            )
        )
        sc_val_2 = eigenvalues[1]
        fig.add_trace(
            go.Scatter3d(
                x=[-sc_val_2 * eigenvectors[1][0], sc_val_2 * eigenvectors[1][0]],
                y=[-sc_val_2 * eigenvectors[1][1], sc_val_2 * eigenvectors[1][1]],
                z=[-sc_val_2 * eigenvectors[1][2], sc_val_2 * eigenvectors[1][2]],
                line=dict(color="green", width=1),
                mode="lines",
                name="PC 2",
            )
        )
        # Set camera position based on eigenvectors
        camera_eye = dict(
            x=2 * eigenvectors[0, 0],  # Scale for better visibility
            y=2 * eigenvectors[0, 1],
            z=2 * eigenvectors[0, 2],
        )

        # Set up the 'up' direction for the camera based on the second eigenvector
        camera_up = dict(x=0 * eigenvectors[0, 1], y=eigenvectors[1, 1], z=0)

        # Update layout with camera settings
        fig.update_layout(scene_camera=dict(eye=camera_eye, up=camera_up))
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(
            df_pca, x="PC 1", y="PC 2", color="target", width=1000, height=700
        )

        st.plotly_chart(fig2)

        with st.sidebar:
            st.markdown("## PCA Results")
            pc_1 = f"{np.round(pca.explained_variance_ratio_[0], 3)*100 :0.1f}%"
            st.metric(label="PC 1 Explained Variance Ratio", value=pc_1)

            pc_2 = f"{np.round(pca.explained_variance_ratio_[1], 3)*100 :0.1f}%"
            st.metric(label="PC 2 Explained Variance Ratio", value=pc_2)

            total_var = f"{np.round(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1], 3)*100 :0.1f}%"
            st.metric(label="Total Explained Variance Ratio", value=total_var)

            reset_ = st.button("Reset")
            if reset_:
                main()


if __name__ == "__main__":
    main()
