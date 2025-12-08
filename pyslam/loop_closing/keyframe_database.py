"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

from collections import defaultdict
from threading import Lock

from pyslam.slam import KeyFrame, Frame


# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.slam.keyframe import KeyFrame
    from pyslam.slam.frame import Frame


class KeyFrameDatabaseDBow(object):
    def __init__(self, voc=None):
        self.voc = voc
        self.inverted_file = defaultdict(list[KeyFrame])  # Similar to mvInvertedFile
        self.mutex = Lock()

    def add(self, keyframe):
        with self.mutex:  # Locking for thread safety
            for word_id, _ in keyframe.g_des:
                self.inverted_file[word_id].append(keyframe)

    def erase(self, keyframe: KeyFrame):
        with self.mutex:  # Locking for thread safety
            for word_id, _ in keyframe.g_des:
                kf_list = self.inverted_file[word_id]
                if keyframe in kf_list:
                    kf_list.remove(keyframe)

    def clear(self):
        with self.mutex:  # Locking for thread safety
            self.inverted_file.clear()
            if self.voc:
                for i in range(self.voc.size()):
                    self.inverted_file[i] = []

    def detect_loop_candidates(self, keyframe: KeyFrame, min_score):
        sp_connected_keyframes = keyframe.get_connected_keyframes()
        l_kfs_sharing_words = [KeyFrame]

        # Search all keyframes that share a word with current keyframes
        # Discard keyframes connected to the query keyframe
        with self.mutex:
            for word_id, _ in keyframe.g_des:
                kf_list = self.inverted_file[word_id]
                for p_kf in kf_list:
                    if p_kf.loop_query_id != keyframe.id:
                        p_kf.num_loop_words = 0
                        if p_kf not in sp_connected_keyframes:
                            p_kf.loop_query_id = keyframe.id
                            l_kfs_sharing_words.append(p_kf)
                    p_kf.num_loop_words += 1

        if not l_kfs_sharing_words:
            return []

        max_common_words = max(p_kf.num_loop_words for p_kf in l_kfs_sharing_words)
        min_common_words = int(max_common_words * 0.8)

        l_score_and_match = []
        for p_kf in l_kfs_sharing_words:
            if p_kf.num_loop_words > min_common_words:
                si = self.voc.score(keyframe.g_des, p_kf.g_des)
                p_kf.loop_score = si
                if si >= min_score:
                    l_score_and_match.append((si, p_kf))

        if not l_score_and_match:
            return []

        l_acc_score_and_match = []
        best_acc_score = min_score

        # Only compare against those keyframes that share enough words
        for score, p_kf in l_score_and_match:
            vp_neighs = p_kf.get_best_covisible_keyframes(10)
            best_score = score
            acc_score = score
            p_best_kf = p_kf

            for p_kf2 in vp_neighs:
                if p_kf2.loop_query_id == keyframe.id and p_kf2.num_loop_words > min_common_words:
                    acc_score += p_kf2.loop_score
                    if p_kf2.loop_score > best_score:
                        p_best_kf = p_kf2
                        best_score = p_kf2.loop_score

            l_acc_score_and_match.append((acc_score, p_best_kf))
            if acc_score > best_acc_score:
                best_acc_score = acc_score

        # Return all those keyframes with a score higher than 0.75*bestScore
        min_score_to_retain = 0.75 * best_acc_score
        sp_already_added_kf = set()
        loop_candidates = []

        for acc_score, p_kf in l_acc_score_and_match:
            if acc_score > min_score_to_retain and p_kf not in sp_already_added_kf:
                loop_candidates.append(p_kf)
                sp_already_added_kf.add(p_kf)

        return loop_candidates

    def detect_relocalization_candidates(self, frame: Frame):
        l_kfs_sharing_words = []

        with self.mutex:
            for word_id, _ in frame.g_des:
                kf_list = self.inverted_file[word_id]
                for p_kf in kf_list:
                    if p_kf.reloc_query_id != frame.mn_id:
                        p_kf.num_reloc_words = 0
                        p_kf.reloc_query_id = frame.mn_id
                        l_kfs_sharing_words.append(p_kf)
                    p_kf.num_reloc_words += 1

        if not l_kfs_sharing_words:
            return []

        max_common_words = max(p_kf.num_reloc_words for p_kf in l_kfs_sharing_words)
        min_common_words = int(max_common_words * 0.8)

        l_score_and_match = []
        for p_kf in l_kfs_sharing_words:
            if p_kf.num_reloc_words > min_common_words:
                si = self.voc.score(frame.g_des, p_kf.g_des)
                p_kf.m_reloc_score = si
                l_score_and_match.append((si, p_kf))

        if not l_score_and_match:
            return []

        l_acc_score_and_match = []
        best_acc_score = 0

        for score, p_kf in l_score_and_match:
            vp_neighs = p_kf.get_best_covisible_keyframes(10)
            best_score = score
            acc_score = score
            p_best_kf = p_kf

            for p_kf2 in vp_neighs:
                if p_kf2.reloc_query_id == frame.mn_id:
                    acc_score += p_kf2.m_reloc_score
                    if p_kf2.m_reloc_score > best_score:
                        p_best_kf = p_kf2
                        best_score = p_kf2.m_reloc_score

            l_acc_score_and_match.append((acc_score, p_best_kf))
            if acc_score > best_acc_score:
                best_acc_score = acc_score

        min_score_to_retain = 0.75 * best_acc_score
        sp_already_added_kf = set()
        reloc_candidates = []

        for acc_score, p_kf in l_acc_score_and_match:
            if acc_score > min_score_to_retain and p_kf not in sp_already_added_kf:
                reloc_candidates.append(p_kf)
                sp_already_added_kf.add(p_kf)

        return reloc_candidates
