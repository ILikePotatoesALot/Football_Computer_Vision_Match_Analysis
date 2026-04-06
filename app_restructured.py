import streamlit as st
from openai import OpenAI
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import re

# Load API keys
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Create directory structure for storing analyses
ANALYSIS_DIR = Path("analyzed_videos")
ANALYSIS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="⚽ Football Performance Analyzer", layout="wide")
st.title("⚽ Football Player Performance Analyzer")
st.write("AI-powered video analysis with tactical insights")

# ==================== HELPER FUNCTIONS (DEFINED FIRST) ====================

def sanitize_folder_name(match_name):
    """Convert match name to valid folder name"""
    safe_name = re.sub(r'[<>:"/\\|?*]', '', match_name)
    safe_name = safe_name.replace(' ', '_')
    safe_name = safe_name[:100]
    return safe_name if safe_name else "Match"

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

def get_unique_folder_name(base_name):
    """Get unique folder name, adding (1), (2), etc. if exists"""
    safe_base = sanitize_folder_name(base_name)
    folder_path = ANALYSIS_DIR / safe_base
    
    if not folder_path.exists():
        return folder_path
    
    counter = 1
    while True:
        new_name = f"{safe_base}({counter})"
        folder_path = ANALYSIS_DIR / new_name
        if not folder_path.exists():
            return folder_path
        counter += 1

def format_video_stats(raw_stats, fps=30, pitch_length_pixels=1920):
    """Convert raw video stats to readable format for DeepSeek"""
    pixels_per_meter = pitch_length_pixels / 105
    formatted = []
    
    for player_id, data in raw_stats.items():
        # Convert distances
        distance_km = (data.get('total_distance_pixels', 0) / pixels_per_meter) / 1000
        distance_with_ball_m = (data.get('distance_with_ball_pixels', 0) / pixels_per_meter)
        sprint_distance_m = (data.get('sprint_distance_pixels', 0) / pixels_per_meter)
        
        formatted.append({
            'player_id': player_id,
            'team': data.get('team', 'Unknown'),
            # Distance & movement
            'distance_covered_km': round(distance_km, 2),
            'time_on_screen_sec': round(data.get('time_on_screen', 0), 1),
            'activity_percentage': round(data.get('activity_percentage', 0), 1),
            # Ball possession
            'possession_percentage': round(data.get('possession_percentage', 0), 1),
            'touches': data.get('touches', 0),
            'distance_with_ball_m': round(distance_with_ball_m, 1),
            'avg_possession_duration_sec': round(data.get('avg_possession_duration_seconds', 0), 1),
            # Pace/intensity
            'sprint_count': data.get('sprint_count', 0),
            'sprint_distance_m': round(sprint_distance_m, 1),
            'walking_pct': round(data.get('walking_percentage', 0), 1),
            'jogging_pct': round(data.get('jogging_percentage', 0), 1),
            'sprinting_pct': round(data.get('sprinting_percentage', 0), 1),
            # Spatial zones
            'time_defensive_third_sec': round(data.get('time_in_defensive_third', 0), 1),
            'time_middle_third_sec': round(data.get('time_in_middle_third', 0), 1),
            'time_attacking_third_sec': round(data.get('time_in_attacking_third', 0), 1)
        })
    
    return formatted


from prompts import get_thinking_prompt, get_output_prompt, get_audio_prompt
def run_video_analysis(stats, match_name, context_notes, player_mapping):
    """
    Three-stage analysis pipeline using external prompt templates
    """
    
    # Prepare stats with player names
    stats_with_names = []
    for player_stat in stats:
        player_id = player_stat['player_id']
        player_name = player_mapping.get(f"P#{player_id}", f"Player #{player_id}")
        stats_with_names.append({
            **player_stat,
            'player_name': player_name
        })
    
    # Convert to JSON string for prompt
    stats_json = json.dumps(stats_with_names, indent=2)
    
    # ========================================
    # STAGE 1: THINKING AGENT
    # ========================================
    
    thinking_prompt = get_thinking_prompt(match_name, stats_json, context_notes)
    
    print("🧠 Stage 1: Running thinking agent (deep analysis)...")
    
    thinking_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": thinking_prompt}],
        temperature=0.2,
        max_tokens=2500
    )
    
    thinking_output = thinking_response.choices[0].message.content
    print(f"✅ Thinking complete ({len(thinking_output)} chars)")
    
    # ========================================
    # STAGE 2: OUTPUT AGENT
    # ========================================
    
    output_prompt = get_output_prompt(thinking_output)
    
    print("📝 Stage 2: Generating comprehensive written report...")
    
    output_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": output_prompt}],
        temperature=0.2,
        max_tokens=3000
    )
    
    final_report = output_response.choices[0].message.content
    print(f"✅ Report generated ({len(final_report)} chars)")
    
    # ========================================
    # STAGE 3: AUDIO AGENT
    # ========================================
    
    audio_prompt = get_audio_prompt(final_report, match_name)
    
    print("🎙️ Stage 3: Generating audio script (2-minute summary)...")
    
    audio_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": audio_prompt}],
        temperature=0.3,
        max_tokens=600
    )
    
    audio_script = audio_response.choices[0].message.content
    
    # Clean up any remaining artifacts
    audio_script = audio_script.replace("##", "")
    audio_script = audio_script.replace("**", "")
    audio_script = audio_script.replace("#", "number")
    
    word_count = len(audio_script.split())
    print(f"✅ Audio script generated ({word_count} words, ~{word_count/2.5:.0f} seconds)")
    
    return {
        'thinking': thinking_output,
        'final_report': final_report,
        'audio_script': audio_script,
        'audio_ready': True
    }


def save_analysis(folder, data):
    """Save analysis metadata to JSON"""
    metadata_file = folder / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

# ==================== SESSION STATE INITIALIZATION ====================

if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'raw_stats' not in st.session_state:
    st.session_state.raw_stats = None
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'analysis_folder' not in st.session_state:
    st.session_state.analysis_folder = None
if 'analysis_result' not in st.session_state: 
    st.session_state.analysis_result = None
if 'current_match_name' not in st.session_state:  
    st.session_state.current_match_name = None

# ==================== TABS ====================

tab1, tab2 = st.tabs(["📹 Analyze New Video", "📂 Past Analyses"])

# ==================== TAB 1: ANALYZE NEW VIDEO ====================

with tab1:
    st.header("Upload Match Video for Analysis")
    
    # STEP 1: Video Upload
    uploaded_video = st.file_uploader(
        "Upload match video (MP4)",
        type=['mp4'],
        help="Upload a video for YOLOv11 tracking and AI analysis"
    )
    
    if uploaded_video and not st.session_state.video_processed:
        st.video(uploaded_video)
        
        if st.button("🎯 Process Video (Step 1/2)", type="primary", use_container_width=True):
            with st.spinner("⚙️ Processing video with YOLOv11... (this may take a few minutes)"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                
                try:
                    from tracking_utils import process_football_video
                    
                    # Create temp folder
                    temp_folder = ANALYSIS_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    temp_folder.mkdir(exist_ok=True)
                    output_video_path = temp_folder / "annotated_video.mp4"
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total, message):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    processed_path, raw_stats, llm_data = process_football_video(
                        str(tfile.name), 
                        str(output_video_path), 
                        progress_callback=progress_callback
                    )

                    # Store LLM data in session state
                    st.session_state.llm_data = llm_data
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store in session state
                    st.session_state.video_processed = True
                    st.session_state.raw_stats = raw_stats
                    st.session_state.output_video_path = str(output_video_path)
                    st.session_state.analysis_folder = temp_folder
                    
                    st.success("✅ Video processing complete! Now add match details below.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    if os.path.exists(tfile.name):
                        os.unlink(tfile.name)
    
    # STEP 2: Show processed video and collect metadata
    if st.session_state.video_processed and st.session_state.raw_stats:
        st.markdown("---")
        st.subheader("📹 Processed Video")
        video_path = Path(st.session_state.output_video_path)

        if video_path.exists():
            try:
                import time
                time.sleep(0.3)
                
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()

                st.video(video_bytes)
                
            except Exception as e:
                st.error(f"❌ Error loading video: {e}")
                st.info(f"Video path: {video_path}")
                st.info("Try refreshing the page or re-processing the video")
        else:
            st.error(f"❌ Video file not found: {video_path}")
        
        # Show detected players
        with st.expander("📊 Detected Players (Raw Stats)", expanded=False):
            detected_players = list(st.session_state.raw_stats.keys())
            st.info(f"Detected {len(detected_players)} players: {', '.join([f'P#{p}' for p in detected_players])}")
            st.json(st.session_state.raw_stats)

        
        st.markdown("---")
        st.subheader("⚙️ Step 2: Add Match Details & Generate Analysis")

            
        match_name = st.text_input(
                "Match Name", 
                f"Match - {datetime.now().strftime('%Y-%m-%d')}",
                help="This will be used as the folder name"
                )
        
        with st.expander("📝 How to use the Context Box", expanded=False):
            st.markdown("""
            **You can provide:**
            - **Formation:** 4-3-3 vs 4-4-2
            - **Player names:** P#10 = Kevin De Bruyne, P#7 = Ronaldo  
            - **Tactical focus:** Analyze Team A
            - **Key moments:** Goal at 5min, substitution at 60min

            **Player ID Merging** (if a player gets a new ID in crowded areas):

            Format: `Merge P#X, P#Y, P#Z into P#TARGET`

            Example:
            ```
            Analyze Team A Only
            Merge P#12, P#26, P#30 into P#30
            Merge P#15, P#21,P#29 into P#29
            ```

             **You can merge as many players as needed!** Just list them with commas.

             **Why?** Tracking may reassign IDs when players enter crowded areas. 
            Merging combines their stats into one player.
            """)

            
        additional_context = st.text_area(
            "Additional Context & Instructions",
            placeholder="Type your instructions here... (click guide above if needed)",
            height=180,
            help="Click the guide above for examples"
        )
        
        
        # Generate Analysis Button
        if st.button("Generate AI Analysis (Step 2/2)", type="primary", use_container_width=True):
            with st.spinner("Generating AI tactical analysis..."):
                try:
                    #  Parse merge instructions
                    from tracking_utils import parse_merge_instructions, merge_player_stats
                    
                    merge_instructions = parse_merge_instructions(additional_context)
                    
                    #  Merge player stats if requested
                    raw_stats = st.session_state.raw_stats
                    if merge_instructions:
                        merged_stats, merge_log = merge_player_stats(raw_stats, merge_instructions)
                        st.success("✅ Merged players: " + ", ".join(merge_log))
                        raw_stats = merged_stats
                    
                    # Format and analyze
                    formatted_stats = format_video_stats(raw_stats)
                    
                    analysis_result = run_video_analysis(
                        formatted_stats, 
                        match_name, 
                        additional_context, 
                        player_mapping={}
                    )

                    
                    #  SAVE TO SESSION STATE
                    st.session_state.analysis_result = analysis_result
                    st.session_state.current_match_name = match_name
                    
                    # Create final folder with match name
                    final_folder = get_unique_folder_name(match_name)
                    old_video_path = st.session_state.output_video_path
                    st.session_state.analysis_folder.rename(final_folder)
                    
                    # UPDATE VIDEO PATH after folder rename
                    st.session_state.output_video_path = str(final_folder / "annotated_video.mp4")
                    # Save everything
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_analysis(final_folder, {
                            "match_name": match_name,
                            "notes": additional_context,
                            "timestamp": timestamp,
                            "folder_name": final_folder.name,
                            "formatted_stats": formatted_stats,
                            "analysis": analysis_result['final_report'],
                            "thinking": analysis_result['thinking'],
                            "full_analysis_result": analysis_result
                        })
                    
                    
                    st.success("✅ Analysis Complete!")
                    st.info("💡 Scroll down to view your analysis and generate audio narration!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    #  AUDIO SECTION - OUTSIDE THE GENERATE BUTTON 
    if st.session_state.analysis_result:
        st.markdown("---")
        st.markdown("---")
        st.subheader("📄 Tactical Analysis Report")
        st.markdown(st.session_state.analysis_result['final_report'])  
        
        

        
        # Download button for text analysis
        st.download_button(
            "📥 Download Analysis Report",
            data=st.session_state.analysis_result['final_report'],
            file_name=f"{st.session_state.current_match_name}_analysis.txt",
            mime="text/plain"
        )
        

        # Audio narration section
        st.markdown("---")
        st.subheader("🎧 Audio Analysis Narration")
        st.caption("Get a 2-minute audio summary perfect for listening on-the-go")

        if st.button("🎙️ Generate Audio Summary", type="secondary", use_container_width=True):
            with st.spinner("🎙️ Creating audio narration..."):
                try: 
                    audio_text = st.session_state.analysis_result['final_report']

                    removals = ["##", "**", "📈", "🔢", "⚖️", "📊", "🎯", "- "]
                    for item in removals:
                        audio_text = audio_text.replace(item, "")
                    
                    # Check length
                    if len(audio_text) > 2000:
                        st.warning(f"⚠️ Report is {len(audio_text)} chars - truncating for audio budget")
                        audio_text = audio_text[:1800]
                        last_period = audio_text.rfind('.')
                        if last_period > 0:
                            audio_text = audio_text[:last_period + 1]
                    
                    # Show the script that will be read
                    with st.expander("📝 Audio Script Preview", expanded=False):
                        st.write(audio_text)
                        st.caption(f"Length: ~{len(audio_text)} characters (~{len(audio_text.split())} words)")
                    
                    # Generate speech with ElevenLabs
                    from elevenlabs import VoiceSettings
                    from elevenlabs.client import ElevenLabs
                    
                    elevenlabs_client = ElevenLabs(
                        api_key=os.getenv("ELEVENLABS_API_KEY")
                    )
                    
                    audio_generator = elevenlabs_client.text_to_speech.convert(
                        voice_id="pNInz6obpgDQGcFmaJgB",  # Adam voice - professional male
                        output_format="mp3_44100_128",
                        text=audio_text,
                        model_id="eleven_multilingual_v2",
                        voice_settings=VoiceSettings(
                            stability=0.2,
                            similarity_boost=0.75,
                            style=0.2,
                            use_speaker_boost=True
                        )
                    )
                    
                    # Save audio to the analysis folder
                    match_folder = None
                    for folder in ANALYSIS_DIR.glob("*/"):
                        metadata_file = folder / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                meta = json.load(f)
                                if meta.get('match_name') == st.session_state.current_match_name:
                                    match_folder = folder
                                    break
                    
                    if match_folder:
                        audio_file_path = match_folder / "audio_summary.mp3"
                    else:
                        audio_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
                    
                    with open(audio_file_path, 'wb') as audio_file:
                        for chunk in audio_generator:
                            audio_file.write(chunk)
                    
                    # Also save the audio script
                    if match_folder:
                        script_file = match_folder / "audio_script.txt"
                        with open(script_file, 'w', encoding='utf-8') as f:
                            f.write(audio_text)
                    
                    # Display audio player
                    st.success("✅ Audio summary ready! Press play below:")
                    st.audio(str(audio_file_path))
                    
                    # Download button for audio
                    with open(audio_file_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Audio Summary",
                            data=f.read(),
                            file_name=f"{st.session_state.current_match_name}_audio_summary.mp3",
                            mime="audio/mpeg",
                            key="download_audio_summary"
                        )
                    
                    st.info("💡 **Tip:** This audio is generated from the concise report above - perfect for quick reviews!")
                    
                except Exception as e:
                    st.error(f"❌ Audio generation failed: {str(e)}")
                    st.info("Check that ELEVENLABS_API_KEY is set in your .env file")
                    import traceback
                    with st.expander("🐛 Error Details"):
                        st.code(traceback.format_exc())
        
        st.info("💡 Check the 'Past Analyses' tab to view this analysis later!")

        st.markdown("---")
        if st.button("🔄 Analyze Another Video", type="secondary", use_container_width=True):
            # Reset everything for new analysis
            st.session_state.video_processed = False
            st.session_state.raw_stats = None
            st.session_state.output_video_path = None
            st.session_state.analysis_result = None
            st.session_state.current_match_name = None
            st.rerun()

# ==================== TAB 2: PAST ANALYSES ====================

with tab2:
    st.header("Previously Analyzed Matches")
    
    # Get all analysis folders (exclude temp folders)
    analysis_folders = sorted(ANALYSIS_DIR.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)
    analysis_folders = [f for f in analysis_folders if not f.name.startswith("temp_")]
    
    if not analysis_folders:
        st.info("📭 No analyses yet. Upload a video in the 'Analyze New Video' tab to get started!")
    else:
        st.write(f"Found **{len(analysis_folders)}** past analyses")
        
        # Search/filter
        search_term = st.text_input("🔍 Search matches", placeholder="Type match name...")
        
        if search_term:
            filtered_folders = [f for f in analysis_folders if search_term.lower() in f.name.lower()]
        else:
            filtered_folders = analysis_folders
        
        if not filtered_folders:
            st.warning(f"No matches found for '{search_term}'")
        
        # Display each analysis
        for folder in filtered_folders:
            metadata_file = folder / "metadata.json"
            video_file = folder / "annotated_video.mp4"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except json.JSONDecodeError as e:
                    st.warning(f"⚠️ Corrupted metadata for {folder.name}: {e}")
                    continue
                except Exception as e:
                    st.error(f"❌ Error reading {folder.name}: {e}")
                    continue
            else:
                metadata = {}
            
            display_name = metadata.get('match_name', folder.name)
            timestamp = metadata.get('timestamp', 'Unknown date')
            
            with st.expander(f"⚽ {display_name} - {timestamp}", expanded=False):
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    if video_file.exists():
                        st.video(str(video_file))
                    else:
                        st.warning("Video file not found")

                        # ⭐ ADD AUDIO PLAYER HERE
                    audio_file = folder / "audio_summary.mp3"
                    if audio_file.exists():
                        st.markdown("**🎧 Audio Summary**")
                        st.audio(str(audio_file))
                        
                        # Download button for audio
                        with open(audio_file, 'rb') as f:
                            st.download_button(
                                "📥 Download Audio",
                                data=f.read(),
                                file_name=f"{display_name}_audio.mp3",
                                mime="audio/mpeg",
                                key=f"audio_{folder.name}"
                            )
                    else:
                        st.caption("🎙️ No audio summary generated for this match")
                    
                    # Stats preview
                    st.subheader("📊 Stats Summary")
                    if 'formatted_stats' in metadata:
                        stats = metadata['formatted_stats']
                        if isinstance(stats, list) and len(stats) > 0:
                            sorted_stats = sorted(stats, key=lambda x: x.get('distance_covered_km', 0), reverse=True)[:5]
                            for i, player in enumerate(sorted_stats, 1):
                                player_id = player.get('player_id', '?')
                                # Show player name if mapped
                                player_mapping = metadata.get('player_mapping', {})
                                player_label = player_mapping.get(f"P#{player_id}", f"Player #{player_id}")
                                st.metric(
                                    player_label,
                                    f"{player.get('distance_covered_km', 0):.2f} km",
                                    f"{player.get('activity_percentage', 0):.1f}% active"
                                )
                        else:
                            st.json(stats)
                
                with col2:
                    st.subheader("🎯 AI Analysis")
                    analysis_data = metadata.get('full_analysis_result', metadata.get('analysis', 'No analysis available'))
                    if isinstance(analysis_data, dict):
                        st.markdown(analysis_data.get('final_report', 'No analysis available'))
                    else:
                        st.markdown(analysis_data)
                    
                    # Additional info
                    if metadata.get('notes'):
                        st.info(f"**Context:** {metadata['notes']}")
                    
                    # Download button
                    st.download_button(
                        "📥 Download Report",
                        data=metadata.get('analysis', ''),
                        file_name=f"{display_name}_report.txt",
                        key=f"download_{folder.name}"
                    )

# ==================== SIDEBAR ====================

st.sidebar.markdown("""
### 🎯 How It Works

**Step 1: Process Video**
- Upload match footage
- YOLOv11 detects and tracks players
- View annotated video with stats

**Step 2: Add Details & Analyze**
- Name the match
- Map player numbers to names (optional)
- Add tactical context
- Generate AI analysis

**Step 3: Audio Narration**
- Get a 2-minute audio summary
- Listen on-the-go
- Download MP3 file

**Step 4: Review**
- Access all past analyses anytime
- Compare different matches
""")

st.sidebar.caption("🤖 AI-Powered Video Analysis Platform")
