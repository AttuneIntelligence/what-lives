import json, re, os
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import leidenalg as la
import igraph as ig
from sklearn.neighbors import kneighbors_graph
import asyncio
import nest_asyncio
from tqdm.notebook import tqdm
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns
import textwrap
from collections import defaultdict

# nest_asyncio.apply()

from .inference import Inference

class WhatLives:
    def __init__(self, inference=None, model=None, n_max=None):
        ### LOAD INFERENCE CLASS FOR LANGUAGE QUERY
        if not inference:
            self.Inference = Inference()
        else:
            self.Inference = inference

        ### CONCURRENCY LIMIT
        self.semaphore_limit = 99
        
        ### CORRELATION MATRIX HYPERPARAMETERS
        self.n_replicates = 6
        
        ### SOURCE XLSX INGRESS
        self.n_max = n_max   # allow for subset of definitions to be analyzed, or `None`
        self.data_dir = "/workspace/what-lives/data"
        self.definitions_table_path = os.path.join(self.data_dir, "what_lives_definitions.xlsx")
        self.definitions_all = self.xlsx_to_json(self.definitions_table_path)
        
        ### FILTER DEFINITIONS TO REMOVE SUPPLEMENTAL
        self.definitions = [item for item in self.definitions_all if item.get('Supplemental') == False]
        
        ### PARSE LAST NAMES IN IMPORTED TABLE
        self.add_last_names()
        
        ### SET MODEL TO USE FOR CORRELATION AND SEMANTIC ANALYSIS
        if not model:
            self.model = self.Inference.model_config["default_models"]["global_default"]
        else:
            if model not in self.Inference.all_models:
                raise ValueError(f"Provided model `{model}` is not supported. Please provide any of : {self.Inference.all_models}")
            else:
                self.model = model
        print(f"Initialized model for analysis: {self.model}")
        
        ### SETUP PATHS -- AFTER MODEL HAS BEEN INSTANTIATED
        self.output_dir = os.path.join(self.data_dir, "results", self.model)
        self.make_out_dir()
        
    #############    
    ### SETUP ###
    #############
    def make_out_dir(self):
        try:
            os.mkdir(self.output_dir)
        except FileExistsError:
            pass
    
    ############################
    ### READ LIFE DEINITIONS ###
    ############################
    def xlsx_to_json(self,
                     table_name,
                     sheet_name='Sheet1',
                     output_path=None,
                     orient='records'):
        try:
            df = pd.read_excel(table_name, sheet_name=sheet_name)
            df = df.where(pd.notnull(df), None)
            json_data = json.loads(df.to_json(orient=orient, date_format='iso'))
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            if not self.n_max:
                return json_data
            else:
                return json_data[:self.n_max]
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        except ValueError as e:
            raise ValueError(f"Error processing Excel file: {str(e)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")
            
    def add_last_names(self):
        for i in range(len(self.definitions)):
            names = self.definitions[i]["Name"].split()
            self.definitions[i]["Last"] = names[-1]
             
    ############################    
    ### CORRELATION MATRICES ###
    ############################
    ### AVERAGE OF MULTIPLE API CALLS
    async def definition_correlation(self, def1, def2):
        ### FORMAT SYSTEM PROMPT
        prompt = self.Inference._read_prompt_template("definition_correlation")
        prompt = prompt.format(
            def1=def1,
            def2=def2,
        )
        question = "What is the correlation metric between -1.0 and 1.0 for these two definitions? Respond with ONLY a single number!"

        ### CREATE THREE ASYNC COROUTINES
        async def get_single_correlation():
            score, metadata = await self.Inference.acomplete(
                text=question, 
                system_prompt=prompt, 
                numerical=True,
                model=self.model
            )
            return score, metadata["cost"]

        ### EXECUTE THREE CORRELATION ANALYSES CONCURRENTLY
        tasks = [get_single_correlation() for _ in range(self.n_replicates)]
        results = await asyncio.gather(*tasks)
        scores, costs = zip(*results)

        ### RETURN AVERAGE CORRELATION, STD, AND TOTAL COST
        average_score = sum(scores) / len(scores)
        std_score = np.std(scores)
        total_cost = sum(costs)
        return average_score, std_score, total_cost

    async def async_define_correlation_matrix(self):
        ### INITIALIZE NULL MATRICES
        n = len(self.definitions)
        M = np.zeros((n, n))  # For average correlations
        S = np.zeros((n, n))  # For standard deviations
        costs = []

        ### CONCURRENT PAIRWISE METRIC GENERATION
        print("Computing correlation matrices...")
        semaphore = asyncio.Semaphore(self.semaphore_limit)
        async def process_pair(i, j):
            async with semaphore:
                avg, std, cost = await self.definition_correlation(
                    def1=self.definitions[i]['Definition'],
                    def2=self.definitions[j]['Definition']
                )
                return i, j, avg, std, cost

        tasks = []
        for i in range(n):
            for j in range(n):
                tasks.append(process_pair(i, j))

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            i, j, avg, std, cost = await task
            M[i, j] = avg
            S[i, j] = std
            costs.append(cost)

        ### SYMMETRIZE, PLOT, AND RETURN 
        self.plot_correlation_matrix(M, title=f'Definition Correlations - {self.model} - Raw')
        self.plot_correlation_matrix(S, title=f'Definition Correlations Standard Deviation - {self.model} - Raw', 
                                   is_std=True)

        Ms = (M.transpose() + M) * 0.5
        Ss = (S.transpose() + S) * 0.5

        self.plot_correlation_matrix(Ms, title=f'Definition Correlation - {self.model}')
        self.plot_correlation_matrix(Ss, title=f'Definition Correlations Standard Deviations - {self.model}', 
                                   is_std=True)

        total_cost = sum(costs)
        print("--- COMPLETE ---")
        print(f"total cost: ${total_cost:.2f}")
        return Ms, Ss

    def plot_correlation_matrix(self, M, title='Definition Correlations', is_std=False):
        plt.figure(figsize=(11,9))

        # Adjust colormap and range based on what we're plotting
        if is_std:
            vmin, vmax = 0, 0.5  # Adjust max range for std dev as needed
            cmap = 'YlOrRd'  # Different colormap for std dev
            cbar_label = 'Standard Deviation'
        else:
            vmin, vmax = -1, 1
            cmap = 'RdYlGn'
            cbar_label = 'Correlation'

        # Create the heatmap
        im = plt.pcolor(M, vmin=vmin, vmax=vmax, cmap=cmap)

        # Create colorbar with appropriate labels
        cbar = plt.colorbar(im, label=cbar_label, fraction=0.046, pad=0.04)

        if not is_std:
            cbar.ax.text(3.5, 1.0, 'agree', ha='left', va='bottom', fontweight='bold')
            cbar.ax.text(3.5, -1.0, 'disagree', ha='left', va='top', fontweight='bold')
            # cbar.ax.text(3.5, 0.0, 'no relation', ha='left', va='bottom', fontweight='bold')

        ### MAP TO NAMES
        names = [d['Name'] for d in self.definitions]

        # Set tick positions and labels
        tick_positions = np.arange(M.shape[0]) + 0.5
        plt.gca().set_xticks(tick_positions)
        plt.gca().set_yticks(tick_positions)
        plt.gca().set_xticklabels(names, rotation=45, ha='right', fontsize=6)
        plt.gca().set_yticklabels(names, fontsize=6)

        # Set the title
        plt.title(title, fontsize=14, pad=20, fontweight='bold')
        # plt.suptitle(self.model, fontsize=12, y=0.8)

        # Save and display
        plt.tight_layout()
        label_name = re.sub(r'[A-Z\s]+', lambda m: m.group().lower().replace(' ', '_'), title)
        plt.savefig(f'{os.path.join(self.output_dir,label_name)}.png', 
                    dpi=600, bbox_inches='tight')
        plt.show()    
    
    def create_correlation_matrix(self):
        M = asyncio.run(self.async_define_correlation_matrix())
        # self.plot_correlation_matrix(M)
        return M   
    
    #########################################
    ### AGGLOMERATIVE CLUSTERING ANALYSIS ###
    #########################################
    def find_optimal_clusters(self, linkage_matrix, distance_matrix, min_clusters=2, max_clusters=15):
        # Get the distances at each merge in the linkage matrix
        distances = linkage_matrix[:, 2]

        # Calculate acceleration (second derivative) of distances
        acceleration = np.diff(distances, n=2)

        # Find elbow point (maximum acceleration)
        elbow_idx = np.argmax(acceleration) + 2
        elbow_n_clusters = len(distances) - elbow_idx + 2

        # Constrain elbow suggestion to our bounds
        elbow_n_clusters = max(min_clusters, min(elbow_n_clusters, max_clusters))
        print(f"Elbow Clusters: {elbow_n_clusters}")
        return elbow_n_clusters

    ### PRIMARY PLOT TO SHOW SORTED CORRELATION MATRIX WITH SUPERIMPOSED LINKAGE DENDROGRAM
    def plot_clustered_correlation_heatmap(self, correlation_matrix, definitions, filename=None, figsize=(20, 20)):
        # Extract names for labels
        names = [d['Name'] for d in definitions]

        # Makes the algorithm more sensitive to strong correlations
        # # Convert correlation matrix to distance matrix
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix))  # Using correlation-based distance
        distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Ensure perfect symmetry

        # Get condensed form for linkage
        condensed_dist = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering using complete linkage
        linkage_matrix = hierarchy.linkage(
            condensed_dist,
            method='complete',  # Complete linkage for clearer cluster separation
            optimal_ordering=True
        )

        # Determine optimal number of clusters
        n_clusters = self.find_optimal_clusters(linkage_matrix, distance_matrix)
        clusters = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Create color palette once for all cluster-related coloring
        unique_clusters = sorted(np.unique(clusters))
        cluster_colors = sns.color_palette("husl", n_colors=len(unique_clusters))  # husl gives better color separation
        color_map = dict(zip(unique_clusters, cluster_colors))

        # Create figure with custom layout
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(2, 2, width_ratios=[0.2, 1], height_ratios=[0.2, 1])

        # Create link colors based on the clusters
        link_cols = {}
        for i, merge in enumerate(linkage_matrix):
            left = int(merge[0])
            right = int(merge[1])

            # Get clusters for merged elements
            if left < len(names):
                left_cluster = clusters[left]
            else:
                left_cluster = link_cols[left - len(names)]['cluster']

            if right < len(names):
                right_cluster = clusters[right]
            else:
                right_cluster = link_cols[right - len(names)]['cluster']

            # If merging within same cluster, use cluster color
            if left_cluster == right_cluster:
                link_cols[i] = {
                    'color': mcolors.rgb2hex(color_map[left_cluster]),
                    'cluster': left_cluster
                }
            else:
                # For between-cluster merges, use light grey
                link_cols[i] = {
                    'color': '#EEEEEE',
                    'cluster': min(left_cluster, right_cluster)
                }

        # Create dendrogram
        ax_dendrogram = fig.add_subplot(gs[0, 1])

        # Determine link colors before creating dendrogram
        def get_link_color(k):
            # return link_cols[k]['color']
            if k < len(link_cols):
                return link_cols[k]['color']
            return 'lightgrey'  # Default color for links outside clusters

        # Create dendrogram with our custom colors
        dendrogram = hierarchy.dendrogram(
            linkage_matrix,
            labels=names,
            leaf_rotation=0,
            leaf_font_size=0,
            ax=ax_dendrogram,
            link_color_func=get_link_color
        )

        # Get reordering information
        reordered_idx = dendrogram['leaves']

        # Reorder everything
        reordered_corr = correlation_matrix[reordered_idx][:, reordered_idx]
        reordered_names = [names[i] for i in reordered_idx]
        reordered_clusters = clusters[reordered_idx]

        # Clean up dendrogram
        ax_dendrogram.set_xticks([])
        for spine in ax_dendrogram.spines.values():
            spine.set_visible(False)

        # Create heatmap
        ax_heatmap = fig.add_subplot(gs[1, 1])

        # Flip matrix for bottom-left to top-right diagonal
        reordered_corr = np.flipud(reordered_corr)
        reordered_names_reversed = list(reversed(reordered_names))
        reordered_clusters_reversed = list(reversed(reordered_clusters))

        im = ax_heatmap.imshow(
            reordered_corr,
            aspect='auto',
            cmap='RdYlGn',
            vmin=-1,
            vmax=1
        )

        # Set up axes and labels
        ax_heatmap.set_xticks(np.arange(len(reordered_names)))
        ax_heatmap.set_yticks(np.arange(len(reordered_names)))
        ax_heatmap.set_xticklabels(reordered_names, rotation=45, ha='right')
        ax_heatmap.set_yticklabels(reordered_names_reversed)

        # Color code labels and add divider lines between clusters
        prev_cluster_x = None
        prev_cluster_y = None

        for idx, (xlabel, ylabel) in enumerate(zip(ax_heatmap.get_xticklabels(), 
                                                 ax_heatmap.get_yticklabels())):
            xlabel_cluster = reordered_clusters[idx]
            ylabel_cluster = reordered_clusters_reversed[idx]

            # Color the labels
            xlabel.set_color(color_map[xlabel_cluster])
            ylabel.set_color(color_map[ylabel_cluster])
            xlabel.set_fontsize(10)
            ylabel.set_fontsize(10)

            # Add vertical lines between different clusters on x-axis
            if prev_cluster_x is not None and xlabel_cluster != prev_cluster_x:
                ax_heatmap.axvline(x=idx-0.5, color='black', linewidth=0.7)
            prev_cluster_x = xlabel_cluster

            # Add horizontal lines between different clusters on y-axis
            if prev_cluster_y is not None and ylabel_cluster != prev_cluster_y:
                ax_heatmap.axhline(y=idx-0.5, color='black', linewidth=0.7)
            prev_cluster_y = ylabel_cluster

        # Add gridlines
        ax_heatmap.set_xticks(np.arange(-.5, len(reordered_names), 1), minor=True)
        ax_heatmap.set_yticks(np.arange(-.5, len(reordered_names), 1), minor=True)
        ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

        # Adjust layout
        gs.update(wspace=0.02, hspace=0.02)

        # Add title
        plt.suptitle(
            'Clustered Definition Correlations',
            fontsize=16, 
            fontweight='bold',
            y=0.95
        )

        # Final layout adjustment
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])

        # Add saving and final rendering
        if filename:
            fout = os.path.join(self.output_dir, filename)
            plt.savefig(fout, dpi=600, bbox_inches='tight')

        # Force complete rendering of the figure
        fig.canvas.draw()

        # Return everything needed for further analysis
        cluster_assignments = {name: cluster for name, cluster in zip(names, clusters)}
        return fig, ax_heatmap, ax_dendrogram, cluster_assignments, reordered_idx, color_map
    
    #############################
    ### SEMANTIC LLM ANALYSIS ###
    #############################
    def print_cluster_definitions(self, cluster_assignments, definitions, color_map=None):
        # Create a reverse mapping of clusters to names
        clusters_to_names = {}
        for name, cluster in cluster_assignments.items():
            if cluster not in clusters_to_names:
                clusters_to_names[cluster] = []
            clusters_to_names[cluster] = clusters_to_names[cluster] + [name]

        # Create a name to definition mapping for quick lookup
        name_to_def = {d['Name']: d['Definition'] for d in definitions}

        # Print each cluster with its definitions
        print("\nCLUSTER ANALYSIS OF LIFE DEFINITIONS")
        print("====================================\n")

        for cluster_num in sorted(clusters_to_names.keys()):
            # Print cluster header
            print(f"\nCluster {cluster_num}")
            print("-" * 50)

            # Print each definition in the cluster
            for name in sorted(clusters_to_names[cluster_num]):
                print(f"\n{name}:")
                # Wrap the definition text for better readability
                definition = name_to_def[name]
                # Add indentation to wrapped text
                wrapped_def = "\n".join(
                    "    " + line 
                    for line in textwrap.wrap(definition, width=80)
                )
                print(wrapped_def)

            print("\n" + "=" * 50)  # Separator between clusters

    def analyze_clusters(self, cluster_assignments, definitions):
        # Create reverse mapping of clusters to names
        clusters_to_names = {}
        for name, cluster in cluster_assignments.items():
            if cluster not in clusters_to_names:
                clusters_to_names[cluster] = []
            clusters_to_names[cluster] = clusters_to_names[cluster] + [name]

        # Analyze cluster sizes
        cluster_sizes = {cluster: len(names) for cluster, names in clusters_to_names.items()}

        # Calculate basic statistics
        total_items = sum(cluster_sizes.values())
        avg_cluster_size = total_items / len(cluster_sizes)

        # Print analysis
        print("\nCLUSTER STATISTICS")
        print("=================")
        print(f"\nTotal number of clusters: {len(cluster_sizes)}")
        print(f"Average cluster size: {avg_cluster_size:.1f} definitions")
        print("\nCluster sizes:")
        for cluster, size in sorted(cluster_sizes.items()):
            print(f"Cluster {cluster}: {size} definitions ({size/total_items*100:.1f}%)")

        return {
            'n_clusters': len(cluster_sizes),
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': avg_cluster_size,
            'total_items': total_items
        }
    
    async def get_cluster_analysis(self, cluster_assignments, definitions):
        # Create a reverse mapping of clusters to names
        clusters_to_names = {}
        for name, cluster in cluster_assignments.items():
            if cluster not in clusters_to_names:
                clusters_to_names[cluster] = []
            clusters_to_names[cluster] = clusters_to_names[cluster] + [name]

        # Create a name to definition mapping for quick lookup
        name_to_def = {d['Name']: d['Definition'] for d in definitions}

        ### GROUP DEFINITIONS TO CLUSTERS AND CALL CLAUDE
        cluster_analysis = {}
        for cluster_num in tqdm(sorted(clusters_to_names.keys())):
            cluster_definitions = []
            cluster_analysis[cluster_num] = {}

            cluster_names = []
            for name in sorted(clusters_to_names[cluster_num]):   
                definition = name_to_def[name]
                cluster_definitions.append(definition)
                cluster_names.append(name)
            cluster_analysis[cluster_num]['names'] = cluster_names

            ### CREATE A PROMPT STRING FROM THE GROUP'S DEFINITIONS
            definitions_str = '\n\n------\n'.join(cluster_definitions)
            # cluster_analysis[cluster_num]['definitions'] = definitions_str
            
            ### a) GET ANALYSIS
            analysis_template = self.Inference._read_prompt_template("cluster_ideas")
            group_analysis, _ = await self.Inference.acomplete(
                text=definitions_str, 
                system_prompt=analysis_template, 
                model=self.model
            )
            cluster_analysis[cluster_num]['group_analysis'] = group_analysis
            
            ### a) GET CONSENSUS DEFINITION FOR CLUSTER
            consensus_template = self.Inference._read_prompt_template("cluster_consensus")
            consensus_definition, _ = await self.Inference.acomplete(
                text=definitions_str, 
                system_prompt=consensus_template, 
                model=self.model
            )
            cluster_analysis[cluster_num]['consensus_definition'] = consensus_definition

        return cluster_analysis
    
    
    def print_cluster_analysis(self, cluster_analysis, markdown_filename=None):
        # Create a string to store the markdown content
        markdown_content = ""

        for cluster_num in sorted(cluster_analysis.keys()):
            # Print to console
            print(f"\n{'='*80}")
            print(f"CLUSTER {cluster_num}")
            print(f"{'='*80}")
            # Print members
            print("\nMEMBERS:")
            print("-" * 40)
            for name in cluster_analysis[cluster_num]['names']:
                print(f"• {name}")
            # Print consensus
            print("\nCONSENSUS:")
            print("-" * 40)
            print(cluster_analysis[cluster_num]['consensus_definition'])
            # Print analysis
            print("\nANALYSIS:")
            print("-" * 40)
            print(cluster_analysis[cluster_num]['group_analysis'])

            # Add to markdown content
            markdown_content += f"# CLUSTER {cluster_num}\n\n"
            markdown_content += "## MEMBERS\n\n"
            for name in cluster_analysis[cluster_num]['names']:
                markdown_content += f"* {name}\n"
            markdown_content += "\n## CONSENSUS\n\n"
            markdown_content += f"{cluster_analysis[cluster_num]['consensus_definition']}\n\n"
            markdown_content += "## ANALYSIS\n\n"
            markdown_content += f"{cluster_analysis[cluster_num]['group_analysis']}\n\n"
            markdown_content += "---\n\n"

        # Save to markdown file if filename is provided
        if markdown_filename:
            fout = os.path.join(self.output_dir, markdown_filename)
            try:
                with open(fout, 'w') as f:
                    f.write(markdown_content)
                print(f"\nCluster analysis saved to {fout}")
            except Exception as e:
                print(f"\nError saving markdown file: {e}")

    
    #################################
    ### COMPLETE ANALYSIS WRAPPER ###
    #################################
    def analyze_definitions(self):
        ### COMPUTE CORRELATION MATRICES OF RESPONSES
        M, S = self.create_correlation_matrix()
        
        ### GENERATE CLUSTERED MATRIX
        fig, ax_heatmap, ax_dendrogram, cluster_assignments, reordered_idx, color_map = self.plot_clustered_correlation_heatmap(M, self.definitions, filename='clustered_correlations.png')
        plt.show()
        
        ### PERFORM SEMANTIC ANALYSIS
        stats = self.analyze_clusters(cluster_assignments, self.definitions)
        self.print_cluster_definitions(cluster_assignments, self.definitions, color_map)
        analysis = asyncio.run(self.get_cluster_analysis(cluster_assignments, self.definitions))
        self.print_cluster_analysis(analysis, markdown_filename="cluster_semantic_analysis.md")
