/*
 *  LexicalChainModel.cpp
 *
 *  Copyright 2015 by Laura Mascarell. All rights reserved.
 *
 *  This file is part of Docent, a document-level decoder for phrase-based
 *  statistical machine translation.
 *
 *  Docent is free software: you can redistribute it and/or modify it under the
 *  terms of the GNU General Public License as published by the Free Software
 *  Foundation, either version 3 of the License, or (at your option) any later
 *  version.
 *
 *  Docent is distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 *  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 *  details.
 *
 *  You should have received a copy of the GNU General Public License along with
 *  Docent. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef docent_LexicalChainModel_h
#define docent_LexicalChainModel_h


#include "SemanticSpace.h"
#include <string>



class Relation {

 private:
  std::string from_;
  std::string to_; 
  Float score_;

  // Relation() {}

 public:
 
 Relation():
  score_(0) {}
  
  const std::string& getFrom() const {
    return from_;
  }

  const std::string& getTo() const {
    return to_;
  }
 
  void setFrom(std::string from) {
    from_ = from;
  }

  void setTo(std::string to) {
    to_ = to;
  }

  void clearFrom() {
    from_.clear();
  }

  void clearTo() {
    to_.clear();
  }

  const Float getScore() const {
    return score_;
  }

  void setScore(const Float score) {
    score_ = score;
  }

};


class LexicalChainModel : public FeatureFunction {
private:
 Logger logger_;
 SemanticSpace *sspace_;
 // std::map<uint, std::map<uint, std::map<std::string, std::vector<const std::string*> > > > ids_;
 // std::vector<Relation> relations_;
 //std::vector<Node> nodes_;

public:

  LexicalChainModel(const Parameters &params);
  virtual ~LexicalChainModel();

  //static Float lookupSim(std::string w1, std::string w2);
  //Float CoSim(const SemanticSpace::WordVector &w1, const SemanticSpace::WordVector &w2) const;

  virtual State *initDocument(const DocumentState &doc, Scores::iterator sbegin) const;
  virtual StateModifications *estimateScoreUpdate(const DocumentState &doc, const SearchStep &step, const State *state,
						  Scores::const_iterator psbegin, Scores::iterator sbegin) const;
  virtual StateModifications *updateScore(const DocumentState &doc, const SearchStep &step, const State *state,
					  StateModifications *estmods, Scores::const_iterator, Scores::iterator estbegin) const;
  virtual FeatureFunction::State *applyStateModifications(FeatureFunction::State *oldState, FeatureFunction::StateModifications *modif) const;
  
  virtual uint getNumberOfScores() const {
    return 1;
  }
  
  virtual void computeSentenceScores(const DocumentState &doc, uint sentno,
				     const FeatureFunction::State *state, Scores::iterator sbegin) const;

};


#endif
