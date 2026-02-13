"""
GUI Module - Dark-themed interface for the Semantic Search Engine.
Simple, functional design focused on usability.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import os
from typing import Optional
from app.config import (
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    DARK_BG, DARK_FG, DARK_BUTTON_BG, DARK_ENTRY_BG, ACCENT_COLOR,
    EMBEDDING_MODELS, VECTOR_DATABASES, DEFAULT_TOP_K, MAX_TOP_K, DATA_DIR
)
from app.retrieval_engine import RetrievalEngine


class SemanticSearchGUI:
    """
    Dark-themed GUI for the Semantic Search Engine.
    Implements all required features from the assignment.
    """
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=DARK_BG)
        
        # Initialize retrieval engine
        self.engine = RetrievalEngine()
        
        # State variables
        self.selected_data_source = None
        self.is_directory = True
        
        # Setup the GUI
        self._setup_styles()
        self._create_widgets()
    
    def _setup_styles(self):
        """Configure ttk styles for dark theme."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Dark.TButton',
                       background=DARK_BUTTON_BG,
                       foreground=DARK_FG,
                       borderwidth=1,
                       focuscolor='none',
                       padding=10)
        style.map('Dark.TButton',
                 background=[('active', ACCENT_COLOR)])
        
        style.configure('Dark.TLabel',
                       background=DARK_BG,
                       foreground=DARK_FG,
                       padding=5)
        
        style.configure('Dark.TFrame',
                       background=DARK_BG)
        
        style.configure('Dark.TLabelframe',
                       background=DARK_BG,
                       foreground=DARK_FG,
                       borderwidth=1)
        style.configure('Dark.TLabelframe.Label',
                       background=DARK_BG,
                       foreground=ACCENT_COLOR,
                       font=('Arial', 10, 'bold'))
        
        style.configure('Accent.TLabel',
                       background=DARK_BG,
                       foreground=ACCENT_COLOR,
                       font=('Arial', 12, 'bold'))
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame', padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="🔍 Semantic Search Engine",
                               style='Accent.TLabel',
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Create sections
        self._create_data_section(main_frame)
        self._create_config_section(main_frame)
        self._create_search_section(main_frame)
        self._create_results_section(main_frame)
    
    def _create_data_section(self, parent):
        """Task 1: GUI-Based Data Selection."""
        section = ttk.LabelFrame(parent, text="Task 1: Data Selection", 
                                style='Dark.TLabelframe', padding=15)
        section.pack(fill=tk.X, pady=(0, 10))
        
        # Data source selection
        data_frame = ttk.Frame(section, style='Dark.TFrame')
        data_frame.pack(fill=tk.X)
        
        ttk.Button(data_frame, text="Select Directory",
                  style='Dark.TButton',
                  command=self._select_directory).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(data_frame, text="Select Single File",
                  style='Dark.TButton',
                  command=self._select_file).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(data_frame, text="Use Sample Data",
                  style='Dark.TButton',
                  command=self._use_sample_data).pack(side=tk.LEFT, padx=5)
        
        # Data info display
        self.data_info_text = tk.Text(section, height=4, width=80,
                                     bg=DARK_ENTRY_BG, fg=DARK_FG,
                                     font=('Courier', 9), relief=tk.FLAT)
        self.data_info_text.pack(fill=tk.X, pady=(10, 0))
        self.data_info_text.insert('1.0', "No data source selected.")
        self.data_info_text.config(state=tk.DISABLED)
    
    def _create_config_section(self, parent):
        """Task 2: Embedding and Vector Store Configuration."""
        section = ttk.LabelFrame(parent, text="Task 2: Configuration", 
                                style='Dark.TLabelframe', padding=15)
        section.pack(fill=tk.X, pady=(0, 10))
        
        config_frame = ttk.Frame(section, style='Dark.TFrame')
        config_frame.pack(fill=tk.X)
        
        # Embedding model selection
        ttk.Label(config_frame, text="Embedding Model:",
                 style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.embedding_var = tk.StringVar(value=list(EMBEDDING_MODELS.keys())[0])
        embedding_menu = ttk.OptionMenu(config_frame, self.embedding_var,
                                       list(EMBEDDING_MODELS.keys())[0],
                                       *list(EMBEDDING_MODELS.keys()))
        embedding_menu.grid(row=0, column=1, padx=5, sticky=tk.EW)
        
        # Model description
        self.model_desc_label = ttk.Label(config_frame, text="",
                                         style='Dark.TLabel',
                                         font=('Arial', 8))
        self.model_desc_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        self._update_model_description()
        self.embedding_var.trace('w', lambda *args: self._update_model_description())
        
        # Vector database selection
        ttk.Label(config_frame, text="Vector Database:",
                 style='Dark.TLabel').grid(row=2, column=0, sticky=tk.W, padx=5, pady=(10, 0))
        
        self.vector_db_var = tk.StringVar(value=VECTOR_DATABASES[0])
        vector_menu = ttk.OptionMenu(config_frame, self.vector_db_var,
                                     VECTOR_DATABASES[0],
                                     *VECTOR_DATABASES)
        vector_menu.grid(row=2, column=1, padx=5, pady=(10, 0), sticky=tk.EW)
        
        # Setup button
        ttk.Button(section, text="⚙ Setup Pipeline",
                  style='Dark.TButton',
                  command=self._setup_pipeline).pack(pady=(15, 0))
        
        # Status display
        self.status_label = ttk.Label(section, text="Status: Not configured",
                                     style='Dark.TLabel',
                                     font=('Arial', 9, 'italic'))
        self.status_label.pack(pady=(5, 0))
        
        config_frame.columnconfigure(1, weight=1)
    
    def _create_search_section(self, parent):
        """Task 3: Semantic Retrieval."""
        section = ttk.LabelFrame(parent, text="Task 3: Semantic Search", 
                                style='Dark.TLabelframe', padding=15)
        section.pack(fill=tk.X, pady=(0, 10))
        
        # Query input
        ttk.Label(section, text="Enter Query:",
                 style='Dark.TLabel').pack(anchor=tk.W)
        
        self.query_entry = tk.Entry(section, bg=DARK_ENTRY_BG, fg=DARK_FG,
                                   font=('Arial', 11), relief=tk.FLAT,
                                   insertbackground=DARK_FG)
        self.query_entry.pack(fill=tk.X, pady=(5, 10), ipady=8)
        
        # Search controls
        controls_frame = ttk.Frame(section, style='Dark.TFrame')
        controls_frame.pack(fill=tk.X)
        
        ttk.Label(controls_frame, text="Top-K:",
                 style='Dark.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        
        self.topk_var = tk.IntVar(value=DEFAULT_TOP_K)
        topk_spinbox = tk.Spinbox(controls_frame, from_=1, to=MAX_TOP_K,
                                 textvariable=self.topk_var,
                                 bg=DARK_ENTRY_BG, fg=DARK_FG,
                                 font=('Arial', 10), width=5,
                                 relief=tk.FLAT, buttonbackground=DARK_BUTTON_BG)
        topk_spinbox.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(controls_frame, text="🔎 Search",
                  style='Dark.TButton',
                  command=self._perform_search).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="Clear Results",
                  style='Dark.TButton',
                  command=self._clear_results).pack(side=tk.LEFT, padx=5)
    
    def _create_results_section(self, parent):
        """Display search results."""
        section = ttk.LabelFrame(parent, text="Task 3: Search Results", 
                                style='Dark.TLabelframe', padding=15)
        section.pack(fill=tk.BOTH, expand=True)
        
        # Results text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(
            section, height=15, bg=DARK_ENTRY_BG, fg=DARK_FG,
            font=('Courier', 9), relief=tk.FLAT, wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for formatting
        self.results_text.tag_config('rank', foreground=ACCENT_COLOR, font=('Courier', 9, 'bold'))
        self.results_text.tag_config('source', foreground='#4EC9B0', font=('Courier', 9, 'italic'))
        self.results_text.tag_config('score', foreground='#CE9178')
        self.results_text.tag_config('content', foreground=DARK_FG)
    
    # Event Handlers
    
    def _select_directory(self):
        """Select a directory containing documents."""
        directory = filedialog.askdirectory(title="Select Document Directory")
        if directory:
            self.selected_data_source = directory
            self.is_directory = True
            self._display_data_info()
    
    def _select_file(self):
        """Select a single document file."""
        file_path = filedialog.askopenfilename(
            title="Select Document File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if file_path:
            self.selected_data_source = file_path
            self.is_directory = False
            self._display_data_info()
    
    def _use_sample_data(self):
        """Use the sample data directory."""
        if os.path.exists(DATA_DIR):
            self.selected_data_source = DATA_DIR
            self.is_directory = True
            self._display_data_info()
        else:
            messagebox.showerror("Error", "Sample data directory not found!")
    
    def _display_data_info(self):
        """Display information about selected data source."""
        self.data_info_text.config(state=tk.NORMAL)
        self.data_info_text.delete('1.0', tk.END)
        
        if self.is_directory:
            # Count files in directory
            files = [f for f in os.listdir(self.selected_data_source) 
                    if f.endswith('.pdf')]
            total_size = sum(os.path.getsize(os.path.join(self.selected_data_source, f)) 
                           for f in files)
            
            info = f"📁 Directory: {self.selected_data_source}\n"
            info += f"📄 Documents: {len(files)} PDF files\n"
            info += f"💾 Total size: {total_size / (1024*1024):.2f} MB\n"
            info += f"📝 Files: {', '.join(files[:5])}"
            if len(files) > 5:
                info += f" ... (+{len(files)-5} more)"
        else:
            # Single file info
            file_size = os.path.getsize(self.selected_data_source)
            info = f"📄 File: {os.path.basename(self.selected_data_source)}\n"
            info += f"📁 Path: {self.selected_data_source}\n"
            info += f"💾 Size: {file_size / (1024*1024):.2f} MB"
        
        self.data_info_text.insert('1.0', info)
        self.data_info_text.config(state=tk.DISABLED)
    
    def _update_model_description(self):
        """Update embedding model description."""
        model_key = self.embedding_var.get()
        if model_key in EMBEDDING_MODELS:
            desc = EMBEDDING_MODELS[model_key]["description"]
            self.model_desc_label.config(text=f"ℹ {desc}")
    
    def _setup_pipeline(self):
        """Setup the semantic search pipeline."""
        if not self.selected_data_source:
            messagebox.showerror("Error", "Please select a data source first!")
            return
        
        self.status_label.config(text="Status: Setting up pipeline...")
        self.root.update()
        
        try:
            # Setup pipeline with selected configuration
            stats = self.engine.setup_pipeline(
                data_source=self.selected_data_source,
                embedding_model=self.embedding_var.get(),
                vector_db=self.vector_db_var.get(),
                is_directory=self.is_directory
            )
            
            status_msg = f"Status: ✓ Ready | {stats['documents_loaded']} docs, "
            status_msg += f"{stats['chunks_created']} chunks"
            self.status_label.config(text=status_msg, foreground='#4EC9B0')
            
            messagebox.showinfo("Success", "Pipeline setup complete!\nReady for semantic search.")
            
        except Exception as e:
            self.status_label.config(text="Status: ✗ Setup failed", foreground='#F48771')
            messagebox.showerror("Error", f"Setup failed:\n{str(e)}")
    
    def _perform_search(self):
        """Perform semantic search."""
        if not self.engine.is_ready:
            messagebox.showerror("Error", "Pipeline not ready! Please setup first.")
            return
        
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query!")
            return
        
        try:
            top_k = self.topk_var.get()
            results = self.engine.search(query, top_k=top_k, return_scores=True)
            
            self._display_results(query, results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Search failed:\n{str(e)}")
    
    def _display_results(self, query, results):
        """Display search results in formatted manner."""
        self.results_text.delete('1.0', tk.END)
        
        # Header
        header = f"Query: '{query}'\nFound {len(results)} relevant documents:\n"
        header += "=" * 80 + "\n\n"
        self.results_text.insert(tk.END, header)
        
        # Display each result
        for result in results:
            rank_text = f"[Rank {result['rank']}] "
            self.results_text.insert(tk.END, rank_text, 'rank')
            
            source_text = f"Source: {os.path.basename(result['source'])}\n"
            self.results_text.insert(tk.END, source_text, 'source')
            
            score_text = f"Relevance Score: {result['relevance_score']:.4f}\n"
            self.results_text.insert(tk.END, score_text, 'score')
            
            content_preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            content_text = f"Content: {content_preview}\n\n"
            self.results_text.insert(tk.END, content_text, 'content')
            
            self.results_text.insert(tk.END, "-" * 80 + "\n\n")
        
        self.results_text.see('1.0')  # Scroll to top
    
    def _clear_results(self):
        """Clear search results."""
        self.results_text.delete('1.0', tk.END)
        self.query_entry.delete(0, tk.END)


def launch_gui():
    """Launch the GUI application."""
    root = tk.Tk()
    app = SemanticSearchGUI(root)
    root.mainloop()

