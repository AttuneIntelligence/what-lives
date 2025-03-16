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
import datetime
from matplotlib.gridspec import GridSpec

# nest_asyncio.apply()

from .inference import Inference

class WhatLives:
    def __init__(self, inference=None, model=None, n_max=None, semaphore_limit=33, n_replicates=6):
        ### LOAD INFERENCE CLASS FOR LANGUAGE QUERY
        if not inference:
            self.Inference = Inference()
        else:
            self.Inference = inference

        ### CONCURRENCY LIMIT
        self.semaphore_limit = semaphore_limit
        
        ### CORRELATION MATRIX HYPERPARAMETERS
        self.n_replicates = n_replicates
        
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

    async def async_define_correlation_matrix(self, checkpoint_freq=100, resume_from_checkpoint=True):
        ### CREATE PROGRESS FILES FOR CHECKPOINT SAVING
        checkpoint_path = os.path.join(self.output_dir, f"correlation_checkpoint_{self.model}.npz")
        progress_path = os.path.join(self.output_dir, f"correlation_progress_{self.model}.json")

        # Get total number of definitions
        n = len(self.definitions)

        # Initialize matrices and progress tracking
        M = np.zeros((n, n))  # For average correlations
        S = np.zeros((n, n))  # For standard deviations
        costs = []

        # Track which pairs have been computed
        computed_pairs = set()

        # Check if checkpoint exists and load if requested
        if os.path.exists(checkpoint_path) and resume_from_checkpoint:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = np.load(checkpoint_path)
            M = checkpoint['correlation_matrix']
            S = checkpoint['std_matrix']

            if os.path.exists(progress_path):
                with open(progress_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    costs = checkpoint_data.get('costs', [])
                    computed_pairs = set(tuple(pair) for pair in checkpoint_data.get('computed_pairs', []))

            print(f"Resumed with {len(computed_pairs)} pairs already computed")

        # Create list of all pairs to compute
        all_pairs = [(i, j) for i in range(n) for j in range(n)]

        # Filter out already computed pairs
        remaining_pairs = [(i, j) for i, j in all_pairs if (i, j) not in computed_pairs]

        if len(remaining_pairs) == 0:
            print("All correlations already computed. Returning checkpoint data.")
            return M, S

        print(f"Computing {len(remaining_pairs)} remaining correlation pairs...")

        # Set up semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.semaphore_limit)

        async def process_pair(i, j):
            async with semaphore:
                avg, std, cost = await self.definition_correlation(
                    def1=self.definitions[i]['Definition'],
                    def2=self.definitions[j]['Definition']
                )
                return i, j, avg, std, cost

        # Process remaining pairs with checkpointing
        tasks = [process_pair(i, j) for i, j in remaining_pairs]

        # Track how many pairs have been processed since last checkpoint
        pairs_since_checkpoint = 0
        checkpoint_counter = 0

        # Process all tasks with progress tracking
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            i, j, avg, std, cost = await task

            # Update matrices
            M[i, j] = avg
            S[i, j] = std
            costs.append(cost)

            # Track this pair as computed
            computed_pairs.add((i, j))

            # Increment counter for checkpoint
            pairs_since_checkpoint += 1

            # Save checkpoint if needed
            if pairs_since_checkpoint >= checkpoint_freq:
                checkpoint_counter += 1
                # print(f"\nSaving checkpoint {checkpoint_counter}...")

                # Save matrix data
                np.savez(
                    checkpoint_path,
                    correlation_matrix=M,
                    std_matrix=S
                )

                # Save progress metadata
                with open(progress_path, 'w') as f:
                    json.dump({
                        'costs': costs,
                        'computed_pairs': list(map(list, computed_pairs)),
                        'timestamp': str(datetime.datetime.now())
                    }, f)

                pairs_since_checkpoint = 0
                # print(f"Checkpoint saved. Completed {len(computed_pairs)}/{len(all_pairs)} pairs.")

        # Final checkpoint after all processing
        print("Saving final results...")
        np.savez(
            checkpoint_path,
            correlation_matrix=M,
            std_matrix=S
        )

        # Save final progress metadata
        with open(progress_path, 'w') as f:
            json.dump({
                'costs': costs,
                'computed_pairs': list(map(list, computed_pairs)),
                'timestamp': str(datetime.datetime.now()),
                'completed': True
            }, f)

        # Symmetrize, plot, and return
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

        # Save the final matrices to their standard locations
        self.save_correlation_matrix(Ms, matrix_name=f'm_correlation_{self.model}')

        return Ms, Ss


    def create_correlation_matrix(self, checkpoint_freq=100, resume_from_checkpoint=True):
        M, S = asyncio.run(self.async_define_correlation_matrix(
            checkpoint_freq=checkpoint_freq,
            resume_from_checkpoint=resume_from_checkpoint
        ))
        
        ### SAVE CORRELATION MATRIX TO FILE
        self.save_correlation_matrix(M, matrix_name=f'm_correlation_{self.model}')
        
        return M, S
    
    ### CHECKPOINT MANAGEMENT ###
    def get_correlation_status(self):
        # Define paths for checkpoint files
        checkpoint_path = os.path.join(self.output_dir, f"correlation_checkpoint_{self.model}.npz")
        progress_path = os.path.join(self.output_dir, f"correlation_progress_{self.model}.json")

        status = {
            "checkpoint_exists": False,
            "progress_exists": False,
            "total_pairs": len(self.definitions) ** 2,
            "computed_pairs": 0,
            "completion_percentage": 0,
            "total_cost": 0,
            "avg_cost_per_pair": 0,
            "estimated_remaining_cost": 0,
            "last_updated": None
        }

        # Check if checkpoint exists
        if os.path.exists(checkpoint_path):
            status["checkpoint_exists"] = True

            # Try to load the checkpoint to verify it's valid
            try:
                checkpoint = np.load(checkpoint_path)
                status["matrix_shape"] = checkpoint['correlation_matrix'].shape
            except Exception as e:
                status["checkpoint_error"] = str(e)

        # Check if progress file exists
        if os.path.exists(progress_path):
            status["progress_exists"] = True

            try:
                with open(progress_path, 'r') as f:
                    progress_data = json.load(f)

                status["computed_pairs"] = len(progress_data.get('computed_pairs', []))
                status["completion_percentage"] = (status["computed_pairs"] / status["total_pairs"]) * 100
                status["total_cost"] = sum(progress_data.get('costs', []))
                status["last_updated"] = progress_data.get('timestamp', None)

                if status["computed_pairs"] > 0:
                    status["avg_cost_per_pair"] = status["total_cost"] / status["computed_pairs"]
                    remaining_pairs = status["total_pairs"] - status["computed_pairs"]
                    status["estimated_remaining_cost"] = status["avg_cost_per_pair"] * remaining_pairs

                status["completed"] = progress_data.get('completed', False)

            except Exception as e:
                status["progress_error"] = str(e)

        return status

    def print_correlation_status(self):
        status = self.get_correlation_status()

        print(f"\n=== Correlation Matrix Status for {self.model} ===")

        if not status["checkpoint_exists"] and not status["progress_exists"]:
            print("No checkpoints found. Correlation matrix has not been started.")
            print(f"Total pairs to compute: {status['total_pairs']}")
            return

        print(f"Completion: {status['completion_percentage']:.2f}% ({status['computed_pairs']}/{status['total_pairs']} pairs)")
        print(f"Cost so far: ${status['total_cost']:.2f}")

        if status["computed_pairs"] > 0:
            print(f"Average cost per pair: ${status['avg_cost_per_pair']:.4f}")
            print(f"Estimated remaining cost: ${status['estimated_remaining_cost']:.2f}")

        if status.get("completed", False):
            print("Status: COMPLETED")
        else:
            print("Status: IN PROGRESS")

        if status["last_updated"]:
            print(f"Last updated: {status['last_updated']}")

        if "checkpoint_error" in status or "progress_error" in status:
            print("\nWarnings:")
            if "checkpoint_error" in status:
                print(f"- Checkpoint file error: {status['checkpoint_error']}")
            if "progress_error" in status:
                print(f"- Progress file error: {status['progress_error']}")


    def reset_correlation_calculation(self, confirmation=False):
        # Define paths for checkpoint files
        checkpoint_path = os.path.join(self.output_dir, f"correlation_checkpoint_{self.model}.npz")
        progress_path = os.path.join(self.output_dir, f"correlation_progress_{self.model}.json")

        # Check if files exist
        files_exist = os.path.exists(checkpoint_path) or os.path.exists(progress_path)

        if not files_exist:
            print("No checkpoint files found. Nothing to reset.")
            return False

        # Get confirmation if needed
        if not confirmation:
            status = self.get_correlation_status()
            print(f"\n=== Reset Correlation Matrix for {self.model} ===")
            print(f"Completion: {status['completion_percentage']:.2f}% ({status['computed_pairs']}/{status['total_pairs']} pairs)")
            print(f"Cost invested so far: ${status['total_cost']:.2f}")

            confirm = input("\nThis will delete all progress. Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                print("Reset cancelled.")
                return False

        # Delete files
        files_deleted = []

        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                files_deleted.append(checkpoint_path)
            except Exception as e:
                print(f"Error deleting checkpoint file: {e}")

        if os.path.exists(progress_path):
            try:
                os.remove(progress_path)
                files_deleted.append(progress_path)
            except Exception as e:
                print(f"Error deleting progress file: {e}")

        if files_deleted:
            print(f"Reset successful. Deleted: {', '.join(files_deleted)}")
            return True
        else:
            print("Reset failed. No files were deleted.")
            return False

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
    
#     def create_correlation_matrix(self):
#         M, S = asyncio.run(self.async_define_correlation_matrix())
#         # self.plot_correlation_matrix(M)
        
#         ### SAVE CORRELATION MATRIX TO FILE
#         self.save_correlation_matrix(M, matrix_name=f'm_correlation_{self.model}')
        
#         return M, S  
    
    def save_correlation_matrix(self, matrix, matrix_name='correlation_matrix'):
        file_path = os.path.join(self.output_dir, f"{matrix_name}.npy")
        np.save(file_path, matrix)
        print(f"Correlation matrix saved to {file_path}")
        return file_path
    
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
    def plot_clustered_correlation_heatmap(self, correlation_matrix, definitions, filename=None, figsize=(28, 20)):
        # Extract names for labels
        names = [d['Name'] for d in definitions]

        # Convert correlation matrix to distance matrix
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
        cluster_colors = sns.color_palette("husl", n_colors=len(unique_clusters))
        color_map = dict(zip(unique_clusters, cluster_colors))

        # Create figure with custom layout
        fig = plt.figure(figsize=figsize)

        # Create a gridspec with proper spacing
        gs = plt.GridSpec(1, 2, width_ratios=[1, 0.4], wspace=0.03, height_ratios=[1])

        # First, compute dendrogram to get the leaf ordering
        dendrogram_info = hierarchy.dendrogram(
            linkage_matrix,
            no_plot=True,  # Just get the ordering information
        )

        # Get the order of leaves
        reordered_idx = dendrogram_info['leaves']

        # Reorder the correlation matrix and names
        reordered_corr = correlation_matrix[reordered_idx][:, reordered_idx]
        reordered_names = [names[i] for i in reordered_idx]
        reordered_clusters = clusters[reordered_idx]

        # Create heatmap (left side)
        ax_heatmap = fig.add_subplot(gs[0, 0])

        # Plot the heatmap with the original ordering first (will reorient later)
        im = ax_heatmap.imshow(
            reordered_corr,
            aspect='auto',  # Force square cells
            cmap='RdYlGn',
            vmin=-1,
            vmax=1
        )

        # Set up axes and labels for heatmap
        ax_heatmap.set_xticks(np.arange(len(reordered_names)))
        ax_heatmap.set_yticks(np.arange(len(reordered_names)))
        ax_heatmap.set_xticklabels(reordered_names, rotation=45, ha='right', fontsize=9)
        ax_heatmap.set_yticklabels(reordered_names, fontsize=9)

        # Flip the y-axis to match dendrogram orientation
        ax_heatmap.invert_yaxis()

        # Define cluster boundaries in reordered coordinates
        cluster_boundaries = {}
        current_cluster = None
        start_idx = 0

        # Find boundaries for clusters in the x-direction (original order)
        for idx, cluster_id in enumerate(reordered_clusters):
            if current_cluster is None:
                current_cluster = cluster_id
                start_idx = idx
            elif cluster_id != current_cluster:
                cluster_boundaries[current_cluster] = (start_idx, idx - 1)
                current_cluster = cluster_id
                start_idx = idx

        # Add the last cluster
        if current_cluster is not None:
            cluster_boundaries[current_cluster] = (start_idx, len(reordered_clusters) - 1)

        # Color code tick labels and add cluster boundaries
        for cluster_id, (start, end) in cluster_boundaries.items():
            color = color_map[cluster_id]

            # Color the x-axis tick labels
            for i in range(start, end + 1):
                ax_heatmap.get_xticklabels()[i].set_color(mcolors.rgb2hex(color))
                ax_heatmap.get_xticklabels()[i].set_fontweight('bold')

            # Color the y-axis tick labels - matching the flipped orientation
            for i in range(start, end + 1):
                ax_heatmap.get_yticklabels()[i].set_color(mcolors.rgb2hex(color))
                ax_heatmap.get_yticklabels()[i].set_fontweight('bold')

            # Add vertical divider lines
            if start > 0:
                ax_heatmap.axvline(x=start-0.5, color='black', linewidth=2.0)

            # Add horizontal divider lines - matching flipped orientation
            if start > 0:
                ax_heatmap.axhline(y=start-0.5, color='black', linewidth=2.0)

            # Draw rectangle around each cluster in the heatmap
            rect = plt.Rectangle(
                (start - 0.5, start - 0.5),  # Position (x_start, y_start) in flipped coordinates
                end - start + 1,              # Width
                end - start + 1,              # Height
                fill=False,
                edgecolor=mcolors.rgb2hex(color),
                linewidth=4,
                zorder=10
            )
            ax_heatmap.add_patch(rect)

        # Add gridlines to heatmap
        ax_heatmap.set_xticks(np.arange(-.5, len(reordered_names), 1), minor=True)
        ax_heatmap.set_yticks(np.arange(-.5, len(reordered_names), 1), minor=True)
        ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

        # Create link colors for dendrogram
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
                # For between-cluster merges, use dark grey
                link_cols[i] = {
                    'color': '#555555',  # Darker gray for better visibility
                    'cluster': min(left_cluster, right_cluster)
                }

        # Create dendrogram on the right side
        ax_dendrogram = fig.add_subplot(gs[0, 1])

        # Function to determine link colors
        def get_link_color(k):
            if k < len(link_cols):
                return link_cols[k]['color']
            return 'black'  # Black for better visibility

        # Create dendrogram with orientation='right'
        R = hierarchy.dendrogram(
            linkage_matrix,
            orientation='right',       # Orient to right side
            labels=None,               # No labels
            no_labels=True,            # Explicitly no labels
            ax=ax_dendrogram,
            link_color_func=get_link_color,
            above_threshold_color='black'
        )

        # Clean up dendrogram axes
        ax_dendrogram.spines['top'].set_visible(False)
        ax_dendrogram.spines['right'].set_visible(False)
        ax_dendrogram.spines['bottom'].set_visible(False)
        ax_dendrogram.spines['left'].set_visible(False)
        ax_dendrogram.tick_params(bottom=False, left=False)
        ax_dendrogram.set_xticks([])
        ax_dendrogram.set_yticks([])

        # Add title
        plt.suptitle(
            'Clustered Definition Correlations',
            fontsize=20, 
            fontweight='bold',
            y=0.98
        )

        # Final layout adjustment - using subplots_adjust instead of tight_layout
        plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.1)

        # Save figure if filename provided
        if filename:
            fout = os.path.join(self.output_dir, filename)
            plt.savefig(fout, dpi=400, bbox_inches='tight')

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
            consensus_input_text = f"Here are the definitions:\n{definitions_str}\n---\nHere is the thematic analysis conducted on these definitions:\n{group_analysis}\n"
            consensus_definition, _ = await self.Inference.acomplete(
                text=consensus_input_text, 
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
