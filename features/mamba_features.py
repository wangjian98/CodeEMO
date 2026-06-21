"""
Mamba Feature Engineering - Step 1: Data Preprocessing & Event Encoding

Transforms raw IDE event logs into the format required by MAMBAStudentModel:
- Event type one-hot (7 dimensions)
- Time interval since last event (log-normalized)
- Distance to deadline (hours)
- Exercise embedding index
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import pickle
import os


# Event type mapping
EVENT_TYPES = ['focus_gained', 'focus_lost', 'text_insert', 'text_remove', 'text_paste', 'run', 'submit']
EVENT_TO_IDX = {et: i for i, et in enumerate(EVENT_TYPES)}


def load_and_preprocess_data(ide_logs_path, passed_path, cache_dir=None):
    """
    Load IDE logs and merge with pass/fail labels
    
    Returns:
        dict: {student_id: {'events': DataFrame, 'passed': bool}}
    """
    print("Loading IDE logs...")
    df = pd.read_csv(ide_logs_path)
    passed = pd.read_csv(passed_path)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Merge with labels
    df = df.merge(passed, on='student', how='left')
    
    # Group by student
    print(f"Processing {df['student'].nunique()} students...")
    student_data = {}
    
    for student_id, group in tqdm(df.groupby('student'), desc="Processing students"):
        group = group.sort_values('timestamp')
        student_data[student_id] = {
            'events': group,
            'passed': group['passed'].iloc[0]
        }
    
    return student_data


def encode_events(student_df, max_events=None):
    """
    Encode events for a single student according to Step 1.
    
    For each event:
    - event_type: one-hot (7 dims)
    - time_interval: log-normalized gap from previous event
    - deadline_distance: hours until deadline
    - exercise_id: exercise index (for embedding)
    - part_id: course part (1-7)
    
    Returns:
        dict with tensors
    """
    events = student_df.sort_values('timestamp')
    
    timestamps = events['timestamp'].values
    event_types = events['eventType'].values
    exercise_ids = events['exercise'].values
    part_ids = events['part'].values
    deadline_dists = events['timeToDeadline'].values / 3600.0  # Convert to hours
    
    n_events = len(events)
    
    # Truncate if needed
    if max_events and n_events > max_events:
        timestamps = timestamps[-max_events:]
        event_types = event_types[-max_events:]
        exercise_ids = exercise_ids[-max_events:]
        part_ids = part_ids[-max_events:]
        deadline_dists = deadline_dists[-max_events:]
        n_events = max_events
    
    # One-hot event types
    event_type_oh = np.zeros((n_events, len(EVENT_TYPES)), dtype=np.float32)
    for i, et in enumerate(event_types):
        if et in EVENT_TO_IDX:
            event_type_oh[i, EVENT_TO_IDX[et]] = 1.0
    
    # Time intervals (log-normalized)
    time_intervals = np.zeros(n_events, dtype=np.float32)
    for i in range(1, n_events):
        dt = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 's')
        dt = max(dt, 1)  # At least 1 second
        time_intervals[i] = np.log1p(dt)
    
    # Normalize time intervals
    if time_intervals.max() > 0:
        time_intervals = time_intervals / (time_intervals.max() + 1e-8)
    
    # Exercise IDs (1-indexed for embedding, max 30)
    exercise_ids = np.clip(exercise_ids, 1, 30).astype(np.int64)
    
    # Part IDs (1-7)
    part_ids = np.clip(part_ids, 1, 7).astype(np.int64)
    
    # Deadline distance (already in hours, normalize)
    deadline_dists = deadline_dists.astype(np.float32)
    if deadline_dists.max() > 0:
        deadline_dists = deadline_dists / (deadline_dists.max() + 1e-8)
    
    return {
        'event_types': torch.LongTensor(event_type_oh.argmax(axis=1)),  # Indices
        'event_types_oh': torch.FloatTensor(event_type_oh),  # One-hot for analysis
        'time_intervals': torch.FloatTensor(time_intervals),
        'exercise_ids': torch.LongTensor(exercise_ids),
        'part_ids': torch.LongTensor(part_ids),
        'deadline_dists': torch.FloatTensor(deadline_dists),
        'n_events': n_events
    }


def collate_mamba_batch(samples):
    """
    Collate function for DataLoader
    
    samples: list of dicts from encode_events()
    
    Returns:
        batch dict with padded sequences
    """
    batch_size = len(samples)
    
    # Find max sequence length
    max_len = max(s['n_events'] for s in samples)
    max_len = min(max_len, 80000)  # Cap for memory
    
    # Pad sequences
    event_types = []
    time_intervals = []
    exercise_ids = []
    part_ids = []
    deadline_dists = []
    masks = []
    
    for s in samples:
        n = s['n_events']
        if n > max_len:
            # Truncate
            event_types.append(s['event_types'][-max_len:])
            time_intervals.append(s['time_intervals'][-max_len:])
            exercise_ids.append(s['exercise_ids'][-max_len:])
            part_ids.append(s['part_ids'][-max_len:])
            deadline_dists.append(s['deadline_dists'][-max_len:])
            masks.append(torch.ones(max_len))
        else:
            # Pad
            pad_len = max_len - n
            event_types.append(F.pad(s['event_types'], (0, pad_len), value=0))
            time_intervals.append(F.pad(s['time_intervals'], (0, pad_len), value=0))
            exercise_ids.append(F.pad(s['exercise_ids'], (0, pad_len), value=0))
            part_ids.append(F.pad(s['part_ids'], (0, pad_len), value=1))
            deadline_dists.append(F.pad(s['deadline_dists'], (0, pad_len), value=0))
            masks.append(F.pad(torch.ones(n), (0, pad_len), value=0))
    
    return {
        'event_types': torch.stack(event_types),
        'time_intervals': torch.stack(time_intervals),
        'exercise_ids': torch.stack(exercise_ids),
        'part_ids': torch.stack(part_ids),
        'deadline_dists': torch.stack(deadline_dists),
        'mask': torch.stack(masks),
        'n_events': torch.LongTensor([min(s['n_events'], max_len) for s in samples])
    }


class MAMBAFeatureProcessor:
    """
    Complete feature processor for Mamba model
    
    Handles:
    1. Data loading and caching
    2. Event encoding
    3. Dataset creation
    """
    def __init__(self, ide_logs_path, passed_path, cache_dir=None, max_events_per_student=80000):
        self.ide_logs_path = ide_logs_path
        self.passed_path = passed_path
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_events = max_events_per_student
        
        self.student_data = None
        self.encodings = {}
    
    def load_data(self, force_reload=False):
        """Load and preprocess raw data"""
        cache_file = self.cache_dir / 'mamba_data.pkl' if self.cache_dir else None
        
        if cache_file and cache_file.exists() and not force_reload:
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.student_data = pickle.load(f)
            return
        
        self.student_data = load_and_preprocess_data(self.ide_logs_path, self.passed_path)
        
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.student_data, f)
            print(f"Cached data to {cache_file}")
    
    def encode_all_students(self, force_recompute=False):
        """Encode all students' events"""
        cache_file = self.cache_dir / 'mamba_encodings.pkl' if self.cache_dir else None
        
        if cache_file and cache_file.exists() and not force_recompute:
            print(f"Loading cached encodings from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.encodings = pickle.load(f)
            return self.encodings
        
        print("Encoding events for all students...")
        for student_id, data in tqdm(self.student_data.items()):
            self.encodings[student_id] = encode_events(data['events'], self.max_events)
        
        if cache_file:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.encodings, f)
            print(f"Cached encodings to {cache_file}")
        
        return self.encodings
    
    def get_student_labels(self):
        """Get labels for all students"""
        labels = {}
        for student_id, data in self.student_data.items():
            labels[student_id] = {
                'passed': data['passed'],
                'grade': 1.0 if data['passed'] else 0.0  # Binary grade for now
            }
        return labels
    
    def create_dataset(self, student_ids=None):
        """Create PyTorch dataset"""
        from torch.utils.data import Dataset
        
        if student_ids is None:
            student_ids = list(self.encodings.keys())
        
        class MAMBAStudentDataset(Dataset):
            def __init__(self, encodings, labels, student_ids):
                self.encodings = encodings
                self.labels = labels
                self.student_ids = student_ids
            
            def __len__(self):
                return len(self.student_ids)
            
            def __getitem__(self, idx):
                student_id = self.student_ids[idx]
                enc = self.encodings[student_id]
                label = self.labels[student_id]
                
                # Get the label tensor
                grade = torch.FloatTensor([label['grade']])
                risk = torch.LongTensor([0 if label['passed'] else 1])
                
                # Return combined dict
                return {
                    'student_id': student_id,
                    'event_types': enc['event_types'],
                    'time_intervals': enc['time_intervals'],
                    'exercise_ids': enc['exercise_ids'],
                    'part_ids': enc['part_ids'],
                    'deadline_dists': enc['deadline_dists'],
                    'n_events': enc['n_events'],
                    'grade': grade,
                    'risk': risk
                }
        
        labels = self.get_student_labels()
        return MAMBAStudentDataset(self.encodings, labels, student_ids)
    
    def get_data_summary(self):
        """Print summary statistics"""
        n_students = len(self.student_data)
        n_events = [len(d['events']) for d in self.student_data.values()]
        
        print(f"\n{'='*50}")
        print(f"MAMBA Feature Processor Summary")
        print(f"{'='*50}")
        print(f"Students: {n_students}")
        print(f"Total events: {sum(n_events):,}")
        print(f"Events per student: {np.mean(n_events):.0f} ± {np.std(n_events):.0f}")
        print(f"Event types: {EVENT_TYPES}")
        print(f"Max events per student (capped): {self.max_events:,}")
        print(f"{'='*50}\n")


def prepare_mamba_training_data(ide_logs_path, passed_path, cache_dir=None):
    """Quick helper to prepare all training data"""
    processor = MAMBAFeatureProcessor(ide_logs_path, passed_path, cache_dir)
    processor.load_data()
    processor.encode_all_students()
    processor.get_data_summary()
    
    labels = processor.get_student_labels()
    dataset = processor.create_dataset()
    
    return dataset, processor


if __name__ == "__main__":
    # Quick test
    processor = MAMBAFeatureProcessor(
        '/tmp/IDE_logs/IDE_logs.csv',
        '/tmp/IDE_logs/passed.csv',
        cache_dir='/tmp/mamba_cache'
    )
    processor.load_data()
    processor.encode_all_students()
    processor.get_data_summary()
    
    # Test encoding
    sample_id = list(processor.encodings.keys())[0]
    enc = processor.encodings[sample_id]
    print(f"Sample student {sample_id}:")
    print(f"  Events: {enc['n_events']}")
    print(f"  Event types shape: {enc['event_types'].shape}")
    print(f"  Time intervals range: [{enc['time_intervals'].min():.2f}, {enc['time_intervals'].max():.2f}]")
