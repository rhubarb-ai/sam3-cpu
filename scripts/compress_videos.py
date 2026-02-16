#!/usr/bin/env python3
"""
Video Compression Script for SAM3

Compresses video files using gzip/tar.gz compression for efficient storage.
Preserves original files and creates compressed archives.

Usage:
    python scripts/compress_videos.py assets/videos/
    python scripts/compress_videos.py video.mp4 --output compressed/
    python scripts/compress_videos.py --batch videos/ --delete-originals
"""

import argparse
import gzip
import tarfile
import shutil
from pathlib import Path
import sys
from typing import Optional


def compress_file_gz(input_path: Path, output_path: Path) -> bool:
    """
    Compress a single file using gzip.
    
    Args:
        input_path: Path to input file
        output_path: Path to output .gz file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"✗ Failed to compress {input_path.name}: {e}")
        return False


def compress_directory_tar(input_dir: Path, output_path: Path) -> bool:
    """
    Compress a directory using tar.gz.
    
    Args:
        input_dir: Path to input directory
        output_path: Path to output .tar.gz file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(input_dir, arcname=input_dir.name)
        return True
    except Exception as e:
        print(f"✗ Failed to compress directory {input_dir.name}: {e}")
        return False


def get_compression_stats(original_path: Path, compressed_path: Path) -> dict:
    """Get compression statistics"""
    original_size = original_path.stat().st_size
    compressed_size = compressed_path.stat().st_size
    ratio = (1 - compressed_size / original_size) * 100
    
    return {
        'original_size_mb': original_size / (1024**2),
        'compressed_size_mb': compressed_size / (1024**2),
        'compression_ratio': ratio,
        'space_saved_mb': (original_size - compressed_size) / (1024**2)
    }


def compress_videos(
    input_path: str,
    output_dir: Optional[str] = None,
    delete_originals: bool = False,
    recursive: bool = False,
    verbose: bool = True
):
    """
    Compress video files or directories.
    
    Args:
        input_path: Path to video file or directory
        output_dir: Output directory (defaults to same as input)
        delete_originals: Delete original files after compression
        recursive: Recursively process directories
        verbose: Print detailed output
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"✗ Error: {input_path} does not exist")
        return
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path.parent if input_path.is_file() else input_path
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    
    # Process single file
    if input_path.is_file():
        if input_path.suffix.lower() in video_extensions:
            output_file = output_path / f"{input_path.name}.gz"
            
            if verbose:
                print(f"Compressing: {input_path.name}")
            
            if compress_file_gz(input_path, output_file):
                stats = get_compression_stats(input_path, output_file)
                
                if verbose:
                    print(f"  Original : {stats['original_size_mb']:.2f} MB")
                    print(f"  Compressed: {stats['compressed_size_mb']:.2f} MB")
                    print(f"  Ratio    : {stats['compression_ratio']:.1f}% reduction")
                    print(f"  Saved    : {stats['space_saved_mb']:.2f} MB")
                    print(f"✓ Saved to: {output_file}")
                
                if delete_originals:
                    input_path.unlink()
                    if verbose:
                        print(f"  Deleted original: {input_path.name}")
        else:
            print(f"⚠ Skipping non-video file: {input_path.name}")
    
    # Process directory
    elif input_path.is_dir():
        # Find all video files
        if recursive:
            video_files = [
                f for f in input_path.rglob("*")
                if f.is_file() and f.suffix.lower() in video_extensions
            ]
        else:
            video_files = [
                f for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() in video_extensions
            ]
        
        if not video_files:
            print(f"⚠ No video files found in {input_path}")
            return
        
        if verbose:
            print(f"{'='*60}")
            print(f"Found {len(video_files)} video file(s) to compress")
            print(f"{'='*60}\n")
        
        total_original = 0
        total_compressed = 0
        successful = 0
        
        for video_file in video_files:
            if verbose:
                print(f"Compressing: {video_file.name}")
            
            # Preserve directory structure in output
            if recursive:
                rel_path = video_file.relative_to(input_path)
                out_dir = output_path / rel_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                output_file = out_dir / f"{video_file.name}.gz"
            else:
                output_file = output_path / f"{video_file.name}.gz"
            
            if compress_file_gz(video_file, output_file):
                stats = get_compression_stats(video_file, output_file)
                total_original += stats['original_size_mb']
                total_compressed += stats['compressed_size_mb']
                successful += 1
                
                if verbose:
                    print(f"  Original : {stats['original_size_mb']:.2f} MB")
                    print(f"  Compressed: {stats['compressed_size_mb']:.2f} MB")
                    print(f"  Ratio    : {stats['compression_ratio']:.1f}% reduction")
                    print(f"✓ Saved to: {output_file}\n")
                
                if delete_originals:
                    video_file.unlink()
                    if verbose:
                        print(f"  Deleted original: {video_file.name}\n")
        
        # Summary
        if verbose:
            print(f"{'='*60}")
            print(f"Compression Summary")
            print(f"{'='*60}")
            print(f"Files processed : {successful}/{len(video_files)}")
            print(f"Total original  : {total_original:.2f} MB")
            print(f"Total compressed: {total_compressed:.2f} MB")
            print(f"Space saved     : {total_original - total_compressed:.2f} MB")
            if total_original > 0:
                overall_ratio = (1 - total_compressed / total_original) * 100
                print(f"Overall ratio   : {overall_ratio:.1f}% reduction")
            print(f"{'='*60}")


def decompress_file(input_path: str, output_path: Optional[str] = None):
    """
    Decompress a .gz file.
    
    Args:
        input_path: Path to .gz file
        output_path: Output path (defaults to removing .gz extension)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"✗ Error: {input_path} does not exist")
        return
    
    if not input_path.suffix == '.gz':
        print(f"✗ Error: {input_path} is not a .gz file")
        return
    
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = input_path.with_suffix('')
    
    print(f"Decompressing: {input_path.name}")
    
    try:
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Decompressed to: {output_path}")
    except Exception as e:
        print(f"✗ Failed to decompress: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compress video files using gzip compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress single video
  python compress_videos.py video.mp4
  
  # Compress all videos in directory
  python compress_videos.py videos/
  
  # Compress with output directory
  python compress_videos.py video.mp4 --output compressed/
  
  # Compress and delete originals
  python compress_videos.py videos/ --delete-originals
  
  # Recursive compression
  python compress_videos.py videos/ --recursive
  
  # Decompress
  python compress_videos.py video.mp4.gz --decompress
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input video file or directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory (default: same as input)"
    )
    
    parser.add_argument(
        "--delete-originals", "-d",
        action="store_true",
        help="Delete original files after compression"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively process directories"
    )
    
    parser.add_argument(
        "--decompress", "-x",
        action="store_true",
        help="Decompress .gz file instead of compressing"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        if args.decompress:
            decompress_file(args.input, args.output)
        else:
            compress_videos(
                input_path=args.input,
                output_dir=args.output,
                delete_originals=args.delete_originals,
                recursive=args.recursive,
                verbose=not args.quiet
            )
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
