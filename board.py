"""
board.py
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
from typing import List, Tuple

from board_base import (
    board_array_size,
    coord_to_point,
    is_black_white,
    is_black_white_empty,
    opponent,
    where1d,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    MAXSIZE,
    NO_POINT,
    PASS,
    GO_COLOR,
    GO_POINT,
)


"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See coord_to_point for explanations of the array encoding.
"""
class GoBoard(object):
    def __init__(self, size: int) -> None:
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.examine_rows_cols_diags()


    def examine_rows_cols_diags(self):
        #Basically this functions saves the entire board by dividing it into rows, columns and diagonals,
        #this will be done by nested loop till it reached the border.
        #The function will save the rows, columns and diagonals in a list.
        #The function will be called after every move to check if there is a 5 in a row.
        if self.size < 5:
            return
        # Calculate all rows, cols, and diags, examine 5-in-a-row
        self.cols = []
        self.rows = []
        for i in range(1, self.size + 1):
            current_row = []
            start = self.row_start(i)
            for pt in range(start, start + self.size):
                current_row.append(pt)
            self.rows.append(current_row)
            
            start = self.row_start(1) + i - 1
            current_col = []
            for pt in range(start, self.row_start(self.size) + i, self.NS):
                current_col.append(pt)
            self.cols.append(current_col)
        
        # Diagnal towards SE direction, from first row (1,1) to (1,n)
        self.diags = []
        start = self.row_start(1)
        for i in range(start, start + self.size):
            diag_SE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)

        # Diagnal towards SE and NE direction, starting from (2,1) to (n,1)
        for i in range(start + self.NS, self.row_start(self.size) + 1, self.NS):
            diag_SE = []
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_SE) >= 5:
                self.diags.append(diag_SE)
            if len(diag_NE) >= 5:
                self.diags.append(diag_NE)

        # diagnal towards NE direction, starting from (n,2) to (n,n)
        start = self.row_start(self.size) + 1
        for i in range(start, start + self.size):
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_NE) >=5:
                self.diags.append(diag_NE)
        assert len(self.rows) == self.size
        assert len(self.cols) == self.size
        assert len(self.diags) == (2 * (self.size - 5) + 1) * 2

    def reset(self, size: int) -> None:
        """
        Creates a start state, an empty board with given size.
        """
        self.blackscore = 0
        self.whitescore = 0
        self.draw = False
        self.winner = EMPTY
        self.size: int = size
        self.NS: int = size + 1
        self.WE: int = 1
        self.ko_recapture: GO_POINT = NO_POINT
        self.last_move: GO_POINT = NO_POINT
        self.last2_move: GO_POINT = NO_POINT
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.examine_rows_cols_diags()

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def _is_legal_check_simple_cases(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check the simple cases of illegal moves.
        Some "really bad" arguments will just trigger an assertion.
        If this function returns False: move is definitely illegal
        If this function returns True: still need to check more
        complicated cases such as suicide.
        """
        assert is_black_white(color)
        if point == PASS:
            return True
        if self.end_of_game():
            return False
        # Could just return False for out-of-bounds, 
        # but it is better to know if this is called with an illegal point
        assert self.pt(1, 1) <= point <= self.pt(self.size, self.size)
        assert is_black_white_empty(self.board[point])
        if self.board[point] != EMPTY:
            return False
        return True

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        if point == PASS:
            return True
        board_copy: GoBoard = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move

    def end_of_game(self) -> bool:
        if self.get_empty_points().size == 0:
            return True
        elif self.blackscore >= 10 or self.whitescore >= 10:
            return True
        elif self.detect_five() != EMPTY:
            return True
        else:
            return False
            
    def get_empty_points(self) -> np.ndarray:
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board_array: np.ndarray) -> None:
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start : start + self.size] = EMPTY

    def is_eye(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block: np.ndarray) -> bool:
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone: GO_POINT) -> np.ndarray:
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        color: GO_COLOR = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point: GO_POINT) -> np.ndarray:
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=np.bool_)
        pointstack = [point]
        color: GO_COLOR = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point: GO_POINT) -> GO_POINT:
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
        and returns NO_POINT otherwise.
        This result is used in play_move to check for possible ko
        """
        single_capture: GO_POINT = NO_POINT
        opp_block = self._block_of(nb_point)
        if not self._has_liberty(opp_block):
            captures = list(where1d(opp_block))
            self.board[captures] = EMPTY
            if len(captures) == 1:
                single_capture = nb_point
        return single_capture

    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Play a move of color on point
        Returns whether move was legal
        """
        if not self._is_legal_check_simple_cases(point, color):
            return False
        # Special cases
        if point == PASS:
            self.ko_recapture = NO_POINT
            self.current_player = opponent(color)
            self.last2_move = self.last_move
            self.last_move = point
            return True

        # General case: deal with captures
        # !!!! IMPLEMENT NEW CAPTURE FOR NINUKI

         # VERTICAL + HORIZ CAPTURES
        opp_color = opponent(color)
        self.board[point] = color

        # row_col_captures(neighbours_list, self.board, color, opp_color)

        points_to_capture = []
        nb_list = self._neighbors(point)
        for nb in nb_list:
            if self.get_color(nb) == opp_color:
                new_idx = nb_list.index(nb)
                new_nb_list = self._neighbors(nb)
                points_to_capture.append(nb_list[new_idx])

                # Check for the next neighbor of the same color
                if self.get_color(new_nb_list[new_idx]) == opp_color:
                    sec_nb_list = self._neighbors(new_nb_list[new_idx])
                    points_to_capture.append(new_nb_list[new_idx])

                    # Check if the second neighbor has the same color as the current player

                    if self.get_color(sec_nb_list[new_idx]) == color:
                        # Process capture
                        # points_to_capture = [new_nb_list[new_idx], neighbours_list[new_idx]]
                        #print(("points to be captured: {}".format(points_to_capture)))
                        #TODO: process capture 
                        
                        self.board[points_to_capture] = EMPTY # make captured point empty
                        if color == WHITE:
                            self.whitescore += len(points_to_capture)
                        elif color == BLACK:
                            self.blackscore += len(points_to_capture)
                           
                                

                # print(points_to_capture)

        # opp_color = opponent(color)
        # in_enemy_eye = self._is_surrounded(point, opp_color)
        # single_captures = []
        # neighbors = self._neighbors(point)
        # for nb in neighbors:
        #     if self.board[nb] == opp_color:
        #         single_capture = self._detect_and_process_capture(nb)
        #         if single_capture != NO_POINT:
        #             single_captures.append(single_capture)
        # block = self._block_of(point)
        # if not self._has_liberty(block):  # undo suicide move
        #     self.board[point] = EMPTY
        #     return False
        # self.ko_recapture = NO_POINT
        # if in_enemy_eye and len(single_captures) == 1:
        #     self.ko_recapture = single_captures[0]
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        return True

    def neighbors_of_color(self, point: GO_POINT, color: GO_COLOR) -> List:
        """ List of neighbors of point of given color """
        nbc: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point: GO_POINT) -> List:
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point: GO_POINT) -> List:
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1,
                point - self.NS + 1,
                point + self.NS - 1,
                point + self.NS + 1]

    def last_board_moves(self) -> List:
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not NO_POINT, not PASS).
        """
        board_moves: List[GO_POINT] = []
        if self.last_move != NO_POINT and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != NO_POINT and self.last2_move != PASS:
            board_moves.append(self.last2_move)
        return board_moves
    
    def detect_five(self):
        """
        Returns black, white or empty if five is detected in any directions
        """
        for r in self.rows:
            result = self.has_five_in_lists(r)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_five_in_lists(c)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_five_in_lists(d)
            if result != EMPTY:
                return result
        return EMPTY
    
    def has_five_in_lists(self, list):
        previous = BORDER
        counter = 1
        for stone in list:
            if self.get_color(stone) == previous:
                counter += 1
            else:
                counter = 1
                previous = self.get_color(stone)
            if counter == 5 and previous != EMPTY:
                return previous
        return EMPTY


