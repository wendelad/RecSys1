// Copyright (C) 2012 Zeno Gantner
// 
// This file is part of MyMediaLite.
// 
// MyMediaLite is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// MyMediaLite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
//  You should have received a copy of the GNU General Public License
//  along with MyMediaLite.  If not, see <http://www.gnu.org/licenses/>.
// 
using System;
using System.Collections.Generic;
using System.Globalization;
using MyMediaLite.Data;
using MyMediaLite.Correlation;
using MyMediaLite.DataType;
using MyMediaLite.Taxonomy;
using System.Linq;

namespace MyMediaLite.RatingPrediction
{
	/// <summary>
	/// Demo user baseline.
	/// </summary>
	public class DemoUserBaseline : BiasedMatrixFactorization, IUserAttributeAwareRecommender, IItemAttributeAwareRecommender
	{
		///
		public IBooleanMatrix UserAttributes
		{
			get { return this.user_attributes; }
			set {
				this.user_attributes = value;
				this.NumUserAttributes = user_attributes.NumberOfColumns;
				this.MaxUserID = Math.Max(MaxUserID, user_attributes.NumberOfRows - 1);
			}
		}
		private IBooleanMatrix user_attributes;
		
		///
		public List<IBooleanMatrix> AdditionalUserAttributes
		{
			get { return this.additional_user_attributes; }
			set {
				this.additional_user_attributes = value;
			}
		}
		private List<IBooleanMatrix> additional_user_attributes;
		
		///
		public int NumUserAttributes { get; private set; }

		///
		public IBooleanMatrix ItemAttributes
		{
			get { return this.item_attributes; }
			set {
				this.item_attributes = value;
				this.NumItemAttributes = item_attributes.NumberOfColumns;
				this.MaxItemID = Math.Max(MaxItemID, item_attributes.NumberOfRows - 1);
			}
		}
		///
		protected IBooleanMatrix item_attributes;
		
		///
		public List<IBooleanMatrix> AdditionalItemAttributes
		{
			get { return this.additional_item_attributes; }
			set {
				this.additional_item_attributes = value;
			}
		}
		private List<IBooleanMatrix> additional_item_attributes;

		///
		public int NumItemAttributes { get; private set; }
		
		/// <summary>Main demographic biases</summary>
		protected float[] main_demo;
		
		/// <summary>Secondary biases</summary>
		protected List<float[]> second_demo;
				
		/// <summary>Main metadata biases</summary>
		protected float[] main_metadata;
		
		/// <summary>Secondary biases</summary>
		protected List<float[]> second_metadata;

		public DemoUserBaseline () : base()
		{		
		}
		
		///
		protected internal override void InitModel()
		{
			base.InitModel();
			
			main_demo = new float[user_attributes.NumberOfColumns];
			second_demo = new List<float[]>(additional_user_attributes.Count);
			for(int d = 0; d < additional_user_attributes.Count; d++)
			{
				float[] element = new float[additional_user_attributes[d].NumberOfColumns];			
				second_demo.Add(element);
			}

			main_metadata = new float[item_attributes.NumberOfColumns];
			second_metadata = new List<float[]>(additional_item_attributes.Count);
			for(int g = 0; g < additional_item_attributes.Count; g++)
			{
				float[] element = new float[additional_item_attributes[g].NumberOfColumns];			
				second_metadata.Add(element);
			}
		}
		
		///
		public override void Train()
		{
			InitModel();
			
			global_bias = ratings.Average;

			for (int current_iter = 0; current_iter < NumIter; current_iter++)
				Iterate();
		}
		
		///
		protected override void Iterate(IList<int> rating_indices, bool update_user, bool update_item)
		{
			float reg = Regularization; // to limit property accesses			
			
			foreach (int index in rating_indices)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];

				float prediction = Predict(u, i, false);
				float err = ratings[index] - prediction;
				
				float user_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByUser[u])) : reg;
				float item_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByItem[i])) : reg;

				// adjust biases
				if (update_user)
					user_bias[u] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * user_reg_weight * user_bias[u]);
				if (update_item)
					item_bias[i] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * item_reg_weight * item_bias[i]);
				
				// adjust user attributes
				if(u < UserAttributes.NumberOfRows)
				{
					IList<int> attribute_list = UserAttributes.GetEntriesByRow(u);
					if(attribute_list.Count > 0)
					{
						foreach (int attribute_id in attribute_list)
						{							
							main_demo[attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * main_demo[attribute_id]);
						}
					}
				}
				
				for(int d = 0; d < AdditionalUserAttributes.Count; d++)
				{
					if(u < AdditionalUserAttributes[d].NumberOfRows)
					{
						IList<int> attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(u);
						if(attribute_list.Count > 0)
						{
							foreach (int attribute_id in attribute_list)
							{
								second_demo[d][attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * second_demo[d][attribute_id]);								
							}
						}
					}	
				}	

				// adjust item attributes
				if(i < ItemAttributes.NumberOfRows)
				{
					IList<int> attribute_list = ItemAttributes.GetEntriesByRow(i);
					if(attribute_list.Count > 0)
					{
						foreach (int attribute_id in attribute_list)
						{							
							main_metadata[attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * main_metadata[attribute_id]);
						}
					}
				}
				
				for(int g = 0; g < AdditionalItemAttributes.Count; g++)
				{
					if(i < AdditionalItemAttributes[g].NumberOfRows)
					{
						IList<int> attribute_list = AdditionalItemAttributes[g].GetEntriesByRow(i);
						if(attribute_list.Count > 0)
						{
							foreach (int attribute_id in attribute_list)
							{
								second_metadata[g][attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * second_metadata[g][attribute_id]);								
							}
						}
					}	
				}
			}

			UpdateLearnRate();
		}
		
		///
		protected override float Predict(int user_id, int item_id, bool bound)
		{						
			double result = global_bias;

			if (user_id < user_bias.Length)
				result += user_bias[user_id];
			if (item_id < item_bias.Length)
				result += item_bias[item_id];
			
			if(user_id < UserAttributes.NumberOfRows)
			{
				IList<int> attribute_list = UserAttributes.GetEntriesByRow(user_id);
				if(attribute_list.Count > 0)
				{
					double sum = 0;
					double second_norm_denominator = attribute_list.Count;
					foreach(int attribute_id in attribute_list) 
					{
						sum += main_demo[attribute_id];
					}
					result += sum / second_norm_denominator;
				}
			}
			
			for(int d = 0; d < AdditionalUserAttributes.Count; d++)
			{
				if(user_id < AdditionalUserAttributes[d].NumberOfRows)
				{
					IList<int> attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(user_id);
					if(attribute_list.Count > 0)
					{
						double sum = 0;
						double second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += second_demo[d][attribute_id];
						}
						result += sum / second_norm_denominator;
					}
				}	
			}

			if(item_id < ItemAttributes.NumberOfRows)
			{
				IList<int> attribute_list = ItemAttributes.GetEntriesByRow(item_id);
				if(attribute_list.Count > 0)
				{
					double sum = 0;
					double second_norm_denominator = attribute_list.Count;
					foreach(int attribute_id in attribute_list) 
					{
						sum += main_metadata[attribute_id];
					}
					result += sum / second_norm_denominator;
				}
			}
			
			for(int g = 0; g < AdditionalItemAttributes.Count; g++)
			{
				if(item_id < AdditionalItemAttributes[g].NumberOfRows)
				{
					IList<int> attribute_list = AdditionalItemAttributes[g].GetEntriesByRow(item_id);
					if(attribute_list.Count > 0)
					{
						double sum = 0;
						double second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += second_metadata[g][attribute_id];
						}
						result += sum / second_norm_denominator;
					}
				}	
			}
			
			if (bound)
			{
				if (result > MaxRating)
					return MaxRating;
				if (result < MinRating)
					return MinRating;
			}
			return (float)result;
		}

		/// <summary>Predict the rating of a given user for a given item</summary>		
		/// <param name="user_id">the user ID</param>
		/// <param name="item_id">the item ID</param>
		/// <returns>the predicted rating</returns>
		public override float Predict(int user_id, int item_id)
		{
			return Predict(user_id, item_id, true);
		}
		
		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"{0} bias_reg={1} reg_u={2} reg_i={3} frequency_regularization={4} learn_rate={5} bias_learn_rate={6} learn_rate_decay={7} num_iter={8} bold_driver={9} ",
				this.GetType().Name, BiasReg, RegU, RegI, FrequencyRegularization, LearnRate, BiasLearnRate, Decay, NumIter, BoldDriver);
		}
	}
}

