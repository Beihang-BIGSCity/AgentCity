#!/usr/bin/env python3
"""
Paper Analysis Script for Top 25 Spatio-Temporal Papers
"""
import pdfplumber
import json
import os
import re
from pathlib import Path

# Define the 25 papers to analyze
PAPERS = [
    {
        "id": 1,
        "title": "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting",
        "venue": "IJCAI",
        "year": 2018,
        "score": 10,
        "datasets": ["METR-LA", "PeMS"],
        "arxiv": "1709.04875",
        "github": None,
        "model_name": "STGCN"
    },
    {
        "id": 2,
        "title": "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting",
        "venue": "ICLR",
        "year": 2018,
        "score": 10,
        "datasets": ["METR-LA", "PeMS-Bay"],
        "arxiv": "1707.01926",
        "github": "https://github.com/liyaguang/DCRNN",
        "model_name": "DCRNN",
        "pdf_exists": True
    },
    {
        "id": 3,
        "title": "Temporal Graph Convolutional Networks for Urban Traffic Flow Prediction Method",
        "venue": "IJCAI",
        "year": 2018,
        "score": 9,
        "datasets": ["PeMS", "Beijing taxi"],
        "arxiv": None,
        "github": None,
        "model_name": "T-GCN"
    },
    {
        "id": 4,
        "title": "Spatio-Temporal Dual Graph Convolutional Network for Traffic Prediction",
        "venue": "IJCAI",
        "year": 2021,
        "score": 9,
        "datasets": ["METR-LA", "PeMS-Bay", "PeMS-SF"],
        "arxiv": None,
        "github": None,
        "model_name": "STDGCN"
    },
    {
        "id": 5,
        "title": "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting",
        "venue": "AAAI",
        "year": 2019,
        "score": 9,
        "datasets": ["METR-LA", "PeMS-Bay"],
        "arxiv": "1803.07066",
        "github": "https://github.com/guoshnBJTU/ASTGCN-r-pytorch",
        "model_name": "ASTGCN",
        "pdf_exists": True
    },
    {
        "id": 6,
        "title": "Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting",
        "venue": "AAAI",
        "year": 2020,
        "score": 9,
        "datasets": ["METR-LA", "PeMS-Bay"],
        "arxiv": "1912.03653",
        "github": "https://github.com/Davidham3/STSGCN",
        "model_name": "STSGCN"
    },
    {
        "id": 7,
        "title": "Dynamic Spatial-Temporal Graph Neural Networks for Traffic Forecasting",
        "venue": "AAAI",
        "year": 2021,
        "score": 9,
        "datasets": ["METR-LA", "PeMS-Bay", "PeMS-08"],
        "arxiv": None,
        "github": None,
        "model_name": "DSTAGNN"
    },
    {
        "id": 8,
        "title": "Multi-Range Attentive Bicomponent Graph Convolutional Network for Traffic Forecasting",
        "venue": "ICLR",
        "year": 2022,
        "score": 9,
        "datasets": ["METR-LA", "PeMS-Bay"],
        "arxiv": "2206.09173",
        "github": "https://github.com/hkuds/MAMGCN",
        "model_name": "MAM-GCN"
    },
    {
        "id": 9,
        "title": "Revisiting Spatial-Temporal Similarity: A Deep Learning Framework for Traffic Prediction",
        "venue": "AAAI",
        "year": 2019,
        "score": 8,
        "datasets": ["METR-LA", "PeMS-Bay", "TaxiBJ"],
        "arxiv": "1803.01254",
        "github": "https://github.com/tangxianfeng/STDN",
        "model_name": "STDN"
    },
    {
        "id": 10,
        "title": "Deep Spatio-temporal Residual Networks for Citywide Crowd Flows Prediction",
        "venue": "AAAI",
        "year": 2016,
        "score": 8,
        "datasets": ["TaxiBJ", "TaxiNYC", "Bike NYC"],
        "arxiv": "1610.00081",
        "github": "https://github.com/lucktroy/DeepST",
        "model_name": "ST-ResNet"
    },
    {
        "id": 11,
        "title": "Deep Spatio-Temporal Convolutional LSTMs for Crowd Flow Prediction",
        "venue": "IJCAI",
        "year": 2017,
        "score": 8,
        "datasets": ["NYC taxi", "Beijing taxi"],
        "arxiv": None,
        "github": None,
        "model_name": "ST-ConvLSTM"
    },
    {
        "id": 12,
        "title": "Exploring Spatio-Temporal Multi-Frequency Analysis for Traffic Flow Forecasting",
        "venue": "ICLR",
        "year": 2022,
        "score": 8,
        "datasets": ["Traffic benchmarks"],
        "arxiv": None,
        "github": None,
        "model_name": "STMFA"
    },
    {
        "id": 13,
        "title": "ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting",
        "venue": "KDD",
        "year": 2021,
        "score": 8,
        "datasets": ["METR-LA", "Traffic"],
        "arxiv": "2107.03045",
        "github": "https://github.com/JLDeng/ST-Norm",
        "model_name": "ST-Norm"
    },
    {
        "id": 14,
        "title": "Contrastive Learning-based Spatial-Temporal Graph Convolutional Networks for Time Series Forecasting",
        "venue": "KDD",
        "year": 2021,
        "score": 8,
        "datasets": ["METR-LA", "PeMS-Bay", "TaxiBJ"],
        "arxiv": None,
        "github": None,
        "model_name": "CL-STGCN"
    },
    {
        "id": 15,
        "title": "Asynchronous Spatio-Temporal Memory Networks for Continuous Event-Based Traffic Flow Prediction",
        "venue": "IJCAI",
        "year": 2020,
        "score": 8,
        "datasets": ["METR-LA", "PeMS-Bay"],
        "arxiv": None,
        "github": None,
        "model_name": "ASTMN"
    },
    {
        "id": 16,
        "title": "DMVST-Net: A Deep Multi-Scale Vector Spatio-Temporal Network for Traffic Forecasting",
        "venue": "AAAI",
        "year": 2018,
        "score": 8,
        "datasets": ["Vehicle trajectory", "METR-LA"],
        "arxiv": None,
        "github": None,
        "model_name": "DMVST-Net"
    },
    {
        "id": 17,
        "title": "Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction",
        "venue": "IJCAI",
        "year": 2017,
        "score": 7,
        "datasets": ["TaxiBJ", "TaxiNYC"],
        "arxiv": None,
        "github": "https://github.com/lucktroy/deepst",
        "model_name": "DeepST"
    },
    {
        "id": 18,
        "title": "Revisiting Recurrent Neural Networks for Robust Traffic Flow Forecasting",
        "venue": "IJCAI",
        "year": 2020,
        "score": 7,
        "datasets": ["METR-LA", "PeMS-Bay"],
        "arxiv": None,
        "github": None,
        "model_name": "RNN-Robust"
    },
    {
        "id": 19,
        "title": "Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting",
        "venue": "IJCAI",
        "year": 2020,
        "score": 7,
        "datasets": ["METR-LA", "PeMS-Bay", "TaxiBJ"],
        "arxiv": None,
        "github": None,
        "model_name": "LDHSTG"
    },
    {
        "id": 20,
        "title": "Spatial-Temporal Heterogeneous Graph Neural Network for Time Series Forecasting",
        "venue": "ICML Workshop",
        "year": 2021,
        "score": 7,
        "datasets": ["METR-LA", "PeMS-Bay"],
        "arxiv": None,
        "github": None,
        "model_name": "ST-HGNN"
    },
    {
        "id": 21,
        "title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
        "venue": "ICLR",
        "year": 2021,
        "score": 7,
        "datasets": ["Multiple time series including traffic"],
        "arxiv": "2012.07436",
        "github": "https://github.com/zhouhaoyi/Informer2020",
        "model_name": "Informer"
    },
    {
        "id": 22,
        "title": "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting",
        "venue": "NeurIPS",
        "year": 2021,
        "score": 7,
        "datasets": ["Traffic benchmarks"],
        "arxiv": "2106.13008",
        "github": "https://github.com/thuml/Autoformer",
        "model_name": "Autoformer"
    },
    {
        "id": 23,
        "title": "Uncertainty Quantification in Deep Learning for Traffic Flow Prediction",
        "venue": "IJCAI Workshop",
        "year": 2020,
        "score": 6,
        "datasets": ["METR-LA", "Traffic"],
        "arxiv": None,
        "github": None,
        "model_name": "UQ-Traffic"
    },
    {
        "id": 24,
        "title": "Graph Attention Networks",
        "venue": "ICLR",
        "year": 2018,
        "score": 6,
        "datasets": ["Graph benchmarks"],
        "arxiv": "1710.10903",
        "github": "https://github.com/PetarV-/GAT",
        "model_name": "GAT"
    },
    {
        "id": 25,
        "title": "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting",
        "venue": "NeurIPS",
        "year": 2015,
        "score": 6,
        "datasets": ["Radar precipitation"],
        "arxiv": "1506.04214",
        "github": None,
        "model_name": "ConvLSTM"
    }
]

def extract_pdf_metadata(pdf_path):
    """Extract metadata and first pages from PDF"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata

            # Extract text from first 3 pages
            text = ""
            for i, page in enumerate(pdf.pages[:3]):
                text += page.extract_text() + "\n"
                if len(text) > 5000:
                    break

            return {
                "metadata": metadata,
                "text_preview": text[:5000],
                "num_pages": len(pdf.pages)
            }
    except Exception as e:
        return {"error": str(e)}

def extract_authors_from_text(text):
    """Extract authors from paper text"""
    # Common patterns for author extraction
    lines = text.split('\n')[:30]  # Check first 30 lines

    # Look for common patterns
    authors = []
    for i, line in enumerate(lines):
        # Skip title and venue lines
        if any(keyword in line.lower() for keyword in ['abstract', 'introduction', 'keywords']):
            break
        # Look for email patterns or affiliation patterns
        if '@' in line or 'university' in line.lower() or 'institute' in line.lower():
            if i > 0 and len(lines[i-1].strip()) > 0:
                potential_authors = lines[i-1].strip()
                if len(potential_authors) < 200:  # Reasonable length
                    authors.append(potential_authors)

    return authors if authors else ["Authors not extracted"]

def main():
    articles_dir = Path("/home/wangwenrui/private/shk/agentCity/data/articles")

    print("Starting analysis of top 25 papers...")
    print("="*80)

    analyzed_papers = []

    for paper in PAPERS:
        print(f"\nAnalyzing Paper {paper['id']}: {paper['title'][:60]}...")

        analysis = {
            "id": paper["id"],
            "title": paper["title"],
            "venue": paper["venue"],
            "year": paper["year"],
            "score": paper["score"],
            "datasets": paper["datasets"],
            "model_name": paper["model_name"],
            "github": paper.get("github"),
            "arxiv": paper.get("arxiv"),
            "pdf_path": None,
            "authors": [],
            "abstract": None,
            "metrics": [],
            "key_findings": []
        }

        # Check if PDF exists locally
        pdf_filename = f"{paper['venue']}{paper['year']}_{paper['model_name']}.pdf"
        pdf_path = articles_dir / pdf_filename

        if pdf_path.exists():
            print(f"  - Found PDF: {pdf_filename}")
            analysis["pdf_path"] = str(pdf_path)

            # Extract information from PDF
            pdf_info = extract_pdf_metadata(str(pdf_path))
            if "error" not in pdf_info:
                text = pdf_info["text_preview"]

                # Try to extract authors
                authors = extract_authors_from_text(text)
                analysis["authors"] = authors

                # Try to extract abstract
                if "abstract" in text.lower():
                    abstract_match = re.search(r'abstract[:\s]+(.*?)(?:introduction|1\s+introduction|keywords)',
                                             text, re.IGNORECASE | re.DOTALL)
                    if abstract_match:
                        analysis["abstract"] = abstract_match.group(1).strip()[:500]

                print(f"  - Extracted {len(authors)} author(s)")
                print(f"  - Pages: {pdf_info['num_pages']}")
        else:
            print(f"  - PDF not found locally")
            if paper.get("arxiv"):
                print(f"  - arXiv ID available: {paper['arxiv']}")

        analyzed_papers.append(analysis)

    # Save analyzed papers
    output_file = articles_dir / "analyzed_top25_papers.json"
    with open(output_file, 'w') as f:
        json.dump(analyzed_papers, f, indent=2)

    print("\n" + "="*80)
    print(f"Analysis complete. Saved to: {output_file}")
    print(f"Total papers analyzed: {len(analyzed_papers)}")
    print(f"Papers with PDFs: {sum(1 for p in analyzed_papers if p['pdf_path'])}")

    # Generate summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'ID':<4} {'Title':<45} {'Venue':<12} {'Year':<6} {'Score':<6} {'PDF':<5}")
    print("-"*80)
    for paper in analyzed_papers:
        title_short = paper['title'][:42] + "..." if len(paper['title']) > 45 else paper['title']
        has_pdf = "Yes" if paper['pdf_path'] else "No"
        print(f"{paper['id']:<4} {title_short:<45} {paper['venue']:<12} {paper['year']:<6} {paper['score']:<6} {has_pdf:<5}")

if __name__ == "__main__":
    main()
