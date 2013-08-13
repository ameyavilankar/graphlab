/*
* \brief Returns the sparse vector containing all the items rated by the user vertex.
*/
rated_type map_get_topk(graph_type::edge_type edge, graph_type::vertex_type other)
{
	// return the list of items rated by the user
	const rated_type& similar_items = other.data().rated_items;

	// To hold the similarity score
	rated_type similarity_score;

	// Get the id of the current item
	id_type current_id = get_other_vertex(edge, other).id();

	// Go through all the items rated by the other user, except for current item
	for(rated_type::const_iterator cit = similar_items.begin(); cit != similar_items.end(); cit++)
		if(cit->first != current_id)
		{
			// Get the score and add only if score is valid
			// Score is invalid if common users are less than MIN_ALLOWED_INTERSECTION			
			double score = adj_cosine_similarity(item_vector[cit->first], item_vector[current_id]);
			if(score != INVALID_SIMILARITY)
				similarity_score[cit->first] = score; 
		}

	return similarity_score;
}

/*
* \brief 
* Aggregates in parallel the list of items rated by each user by map reducing on all the user
* vertices connected to an item. It also counts the number of users common between the current item
* and the items in the aggregated list. It then removes the item that have less than MIN_ALLOWED_INTERSECTION
* common users. 
*/
void get_topk(engine_type::context& context, graph_type::vertex_type vertex)
{
	// Gather the list of items rated by each user.
	rated_type gather_result = graphlab::warp::map_reduce_neighborhood<rated_type, graph_type::vertex_type>(vertex, graphlab::IN_EDGES, map_get_topk, combine);

	// Get a reference to the vertex data
	vertex_data& vdata = vertex.data();

	// Store the list of similar items into the recommended items
	vdata.recommended_items = gather_result;

	// TODO: Trim the result to have only topk entries using a heap

	// Increment the num_updates to the vertex by 1
	vdata.num_updates++;
}